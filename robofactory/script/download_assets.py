from huggingface_hub import snapshot_download
import os

snapshot_download(repo_id='sparklexfantasy/RoboFactory_asset',
                  local_dir='./assets',
                  repo_type='dataset',
                  resume_download=True)