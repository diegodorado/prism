
        @echo off
        echo Current directory: %CD%
        cd /d "D:/SDTD299/StreamDiffusion"
        echo Changed directory to: %CD%
        set "PIP_DISABLE_PIP_VERSION_CHECK=1"

        if exist "venv" (
            echo Clearing pip cache before update...
            python -m pip cache purge
        )

        if not exist "venv" (
            echo Creating Python venv at: "D:/SDTD299/StreamDiffusion\venv"
            "D:/SDTD299/StreamDiffusion\venv\Scripts\python.exe" -m venv venv
        ) else (
            echo Virtual environment already exists at: "D:/SDTD299/StreamDiffusion\venv"
        )

        echo Attempting to activate virtual environment...
        call "venv\Scripts\activate.bat"

        rem Check if the virtual environment was activated successfully
        if "%VIRTUAL_ENV%" == "" (
            echo Failed to activate virtual environment.
            pause
            exit /b 1
        ) else (
            echo Virtual environment activated.
        )


        echo Installing nvidia-pyindex to ensure access to NVIDIA-specific packages...
        python -m pip install  nvidia-pyindex --trusted-host pypi.org --trusted-host files.pythonhosted.org

        echo Installing huggingface_hub==0.24.6 first to prevent version conflicts...
        python -m pip install  huggingface_hub==0.24.6

        echo Installing dependencies with pip from the activated virtual environment...
        python -m pip install  torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
        python -m pip install  -e .
        python setup.py develop
        python -m pip install  -r streamdiffusionTD/requirements_pc.txt

        echo Ensuring huggingface_hub version is correct...
        python -m pip install  --force-reinstall huggingface_hub==0.24.6

        echo Installation Finished
        pause
        
