import truss
import baseten

import os

ENV_VAR_NAME = 'BASETEN_API_KEY'

if ENV_VAR_NAME not in os.environ:
    print("Please set `BASETEN_API_KEY` to login")
    import sys
    sys.exit(1)

baseten.login(os.environ[ENV_VAR_NAME])

gpt_truss = truss.from_directory('./gpt-neox-20b-truss')
baseten.deploy(gpt_truss)
