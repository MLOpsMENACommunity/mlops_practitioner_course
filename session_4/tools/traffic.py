"""Replay ride traffic against the model API, honouring whatever is broken.

    python -m tools.traffic                    # normal, ~20 rps, runs until Ctrl-C
    python -m tools.traffic --rps 40 --seconds 60
    python -m tools.traffic --profile ramadan  # or let incident 06 set it
    python -m tools.traffic --bad-rate 0       # only well-formed requests

The generator is where three incidents actually live, because three of them are
UPSTREAM failures — the API is innocent and its own metrics look fine:

  01  FEED_DISTANCE_IN_MILES   sends miles in a field the API reads as km
  05  NEW_CITY_ROLLOUT         ~8% of trips start arriving from alexandria
  06  TRAFFIC_PROFILE=ramadan  demand moves into the hours before iftar

It also closes the label loop: a share of requests come back later with the
true duration via /feedback, which is what MAE, the drift-vs-accuracy
comparison and incident 15 all depend on.
"""

from __future__ import annotations

import argparse
import random
import time

import httpx

from incidents import state
from services.ride_model import CITY_DISTANCE_RANGE, HOUR_WEIGHTS, true_duration

#: Imported, not redefined: the model was FIT on this same hour distribution
#: (services/ride_model.py). Ramadan moves demand toward the two hours before
#: iftar, and the drift that produces is REAL — which is exactly what makes the
#: naive retrain gate promote a model fitted to three weeks of the year.
MILES_PER_KM = 0.621371

#: Schema violations a real caller actually makes: a field that drifted out of
#: range, a type that arrived as a string, a field that stopped being sent at
#: all. Each one names a DIFFERENT field, which is the whole point of the
#: validation panel — "422s are up" starts an investigation, "422s are up on
#: passengers" ends one.
BAD_PAYLOADS = (
    {"distance_km": -4.0, "passengers": 2, "hour_of_day": 9, "city": "cairo"},
    {"distance_km": 12.0, "passengers": 99, "hour_of_day": 9, "city": "cairo"},
    {"distance_km": 12.0, "passengers": 2, "hour_of_day": 25, "city": "cairo"},
    {"distance_km": "twelve", "passengers": 2, "hour_of_day": 9, "city": "cairo"},
    {"passengers": 2, "hour_of_day": 9, "city": "cairo"},  # distance_km missing
)


def _sample(rng: random.Random, profile: str) -> tuple[float, int, int, str]:
    """Draw one real trip: (distance_km, passengers, hour_of_day, city)."""
    weights = HOUR_WEIGHTS.get(profile, HOUR_WEIGHTS["normal"])
    hour = rng.choices(range(24), weights=weights, k=1)[0]
    city = "cairo"
    if state.flag("NEW_CITY_ROLLOUT") and rng.random() < 0.08:
        city = "alexandria"
    low, high = CITY_DISTANCE_RANGE[city]
    return rng.uniform(low, high), rng.randint(1, 4), hour, city


def send_one(
    client: httpx.Client,
    rng: random.Random,
    profile: str,
    feedback_rate: float,
    bad_rate: float = 0.0,
) -> None:
    """Send one prediction, and sometimes the late label that follows it."""
    # A small share of malformed requests, because real callers send them. Without
    # this the error-rate panel is a flat zero and the validation panel is empty —
    # and a monitoring lab where the error rate CANNOT move teaches the wrong
    # lesson about what a green dashboard means. Set --bad-rate 0 to turn it off.
    if bad_rate and rng.random() < bad_rate:
        try:
            client.post("/predict", json=rng.choice(BAD_PAYLOADS), timeout=5.0)
        except httpx.HTTPError:
            pass
        return

    distance_km, passengers, hour, city = _sample(rng, profile)

    # Incident 01. The upstream feed changed units; the field name did not, so
    # the API happily reads 12.4 miles as 12.4 km and quotes a shorter ride.
    sent_distance = (
        distance_km * MILES_PER_KM if state.flag("FEED_DISTANCE_IN_MILES") else distance_km
    )

    try:
        resp = client.post(
            "/predict",
            json={
                "distance_km": round(sent_distance, 3),
                "passengers": passengers,
                "hour_of_day": hour,
                "city": city,
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        request_id = resp.json()["request_id"]
    except (httpx.HTTPError, KeyError):
        return

    # The label describes the trip that actually happened — in kilometres,
    # in the city it happened in. That gap is what MAE eventually reveals.
    if rng.random() < feedback_rate:
        actual = float(true_duration(distance_km, passengers, hour, city)) + rng.gauss(0.0, 1.0)
        try:
            client.post(
                "/feedback",
                json={"request_id": request_id, "actual_duration_min": round(actual, 2)},
                timeout=5.0,
            )
        except httpx.HTTPError:
            pass


def run(
    base_url: str,
    rps: float,
    seconds: float | None,
    profile: str,
    feedback_rate: float,
    bad_rate: float = 0.0,
) -> None:
    """Drive the API at roughly `rps` until `seconds` elapse (or forever)."""
    rng = random.Random(42)
    interval = 1.0 / rps
    started = time.perf_counter()
    sent = 0
    with httpx.Client(base_url=base_url) as client:
        while seconds is None or (time.perf_counter() - started) < seconds:
            # Re-read every tick: an incident injected mid-run takes effect now.
            active_profile = state.flag("TRAFFIC_PROFILE", profile) or profile
            send_one(client, rng, active_profile, feedback_rate, bad_rate)
            sent += 1
            if sent % 200 == 0:
                print(f"traffic: {sent} requests, profile={active_profile}", flush=True)
            time.sleep(interval)
    print(f"traffic: done, {sent} requests")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="python -m tools.traffic", description=__doc__)
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=None, help="default: run until Ctrl-C")
    parser.add_argument("--profile", default="normal", choices=sorted(HOUR_WEIGHTS))
    parser.add_argument(
        "--feedback-rate",
        type=float,
        default=0.35,
        help="share of rides whose true duration comes back as a late label",
    )
    parser.add_argument(
        "--bad-rate",
        type=float,
        default=0.02,
        help="share of requests that violate the schema (feeds the error-rate and "
        "validation-failure panels); 0 disables them",
    )
    args = parser.parse_args()
    assert set(CITY_DISTANCE_RANGE) >= {"cairo", "alexandria"}, "both segments need a profile"
    try:
        run(args.url, args.rps, args.seconds, args.profile, args.feedback_rate, args.bad_rate)
    except KeyboardInterrupt:
        print("\ntraffic: stopped")


if __name__ == "__main__":
    main()
