# Project_CV_25_26
Computer Vision project about In-home physical therapy pose estimation system

# cosa fare prima di avviare tutto:

# 0. creare il venv se non esiste:
    python -m venv venv

# 1. attivare il venv:
    .\.venv\Scripts\Activate.ps1

    (se dà un errore di sicurezza, usare il codice:

        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

    e poi riaprive il venv con il codice sopra ".\.venv\Scripts....")

# 2. installare le dipendenze:
    pip install ultralytics opencv-python scipy torch pyyaml  ## per l'acquisizione tramite webcam
    pip install scipy torch pyyaml
    pip install fastdtw
