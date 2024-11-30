import sys
import uvicorn
from app_main import app

def main(argv):
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", reload=('--reload' in argv), log_config="log_settings.ini")

if __name__ == '__main__':
    main(sys.argv)