# go back to the exact data + pipeline state of v1
git checkout data-v1
dvc checkout                 # or: dvc pull, if the cache lacks that version

# return to latest
git checkout main
dvc checkout
For the next version
The repeatable recipe going forward:


# after changing data or re-running the pipeline
dvc repro            # or dvc add <file> for raw data
dvc push             # data to remote FIRST
git add dvc.lock data/raw/*.dvc
git commit -m "data: snapshot v2 ..."
git tag -a data-v2 -m "..."
git push && git push origin data-v2
