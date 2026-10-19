#!/usr/bin/env bash
# Train on a Google Cloud Compute Engine VM (free-trial credits are enough; the job takes ~1 min).
# Prereqs: gcloud auth login ; gcloud config set project YOUR_PROJECT_ID
# Usage:   bash scripts/vertex_train.sh https://github.com/YOUR_USERNAME/smart-glove-sim.git
set -euo pipefail
REPO="${1:?pass your GitHub repo URL}"
VM=glove-train; ZONE=asia-south1-a
gcloud services enable compute.googleapis.com
gcloud compute instances create "$VM" --zone="$ZONE" --machine-type=e2-standard-4 \
  --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=20GB
sleep 40   # let SSH come up
gcloud compute ssh "$VM" --zone="$ZONE" --command "
  sudo apt-get update -qq && sudo apt-get install -y -qq git python3-pip python3-venv &&
  git clone $REPO && cd smart-glove-sim && python3 -m venv .venv && . .venv/bin/activate &&
  pip install -q -r requirements.txt && python scripts/train.py"
gcloud compute scp --zone="$ZONE" --recurse "$VM:smart-glove-sim/models" "$VM:smart-glove-sim/outputs" ./
gcloud compute instances delete "$VM" --zone="$ZONE" --quiet    # stop billing
echo "Done: models/ and outputs/ copied back. Commit and push them."
