#!/bin/bash

# Upload files in a folder

FILEPATHS=$(find ./ -name "*.tar.gz")
DROPBOX_PATH="/s3prl/hear_benchmark_datasets/"
ACCESS_TOKEN="sl.BW6Lfc8GOlUxkpQTlpSDjQf5c5N51yUgwmibt69WEc-erWmhdhvG-MEg1yQz-dC1CIsYCWw1HKEZDzCo1nN1icTDft0eUey0jV-HVkpuPMIU9lSsm_oBpnAOls-bcsddYZMuV98"

for FILEPATH in $FILEPATHS; do
    FILE_NAME=$(basename ${FILEPATH})

    split -b 100M $FILE_NAME ${FILE_NAME}.
    SPLITPATHS=$(find ./ -name "${FILE_NAME}.??")

    for SPLITPATH in $SPLITPATHS; do
        SPLIT_NAME=$(basename ${SPLITPATH})

        curl -X POST https://content.dropboxapi.com/2/files/upload  \
            --header "Authorization: Bearer ${ACCESS_TOKEN}" \
            --header "Dropbox-API-Arg: {\"path\": \"${DROPBOX_PATH}${SPLIT_NAME}\"}" \
            --header "Content-Type: application/octet-stream" \
            --data-binary @$SPLITPATH \
            --progress-bar
    done

    rm ${FILE_NAME}.??
done

echo "upload complete"