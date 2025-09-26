# data/google_drive_client.py
import os
import json
import io
import logging
import time
from datetime import datetime

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), '..', 'credentials.json')
TOKEN_FILE = os.path.join(os.path.dirname(__file__), '..', 'token.json')
DEFAULT_SHARE_EMAIL = "kendeabreu24@gmail.com"

ML_GPT_FOLDER_ID = "1Z3uTvxJ-MvBNP1zH7SNR8qI-B__Gg6Lb"  # Pasta para modelos de ML
CSV_FOLDER_ID = "1-2NSyy8C4kuBt_Rb6KKB42CVOxJLzTq8"
PKL_FOLDER_ID = "1-9FzKbCYdYuS2peZ5WlCTdtR7ayJZPH9"

def get_drive_service():
    try:
        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        logging.error(f"Erro ao obter serviço do Google Drive: {e}")
        raise

def share_file_with_user(file_id, user_email=DEFAULT_SHARE_EMAIL):
    try:
        service = get_drive_service()
        permission = {
            'type': 'user',
            'role': 'writer',
            'emailAddress': user_email
        }
        service.permissions().create(fileId=file_id, body=permission, sendNotificationEmail=False).execute()
        print(f"✅ Arquivo compartilhado com {user_email} (ID: {file_id})")
    except Exception as e:
        print(f"⚠️ Falha ao compartilhar arquivo {file_id} com {user_email}: {e}")

def get_folder_id_for_file(filename):
    if filename.lower().endswith('.csv'):
        return CSV_FOLDER_ID
    elif filename.lower().endswith('.pkl'):
        return PKL_FOLDER_ID
    return None

def find_file_id(filename, drive_folder_id=None):
    try:
        service = get_drive_service()
        query = f"name = '{filename}'"
        if drive_folder_id:
            query += f" and '{drive_folder_id}' in parents"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return None
    except Exception as e:
        logging.error(f"Erro ao buscar arquivo {filename} no Drive: {e}")
        return None

def upload_or_update_file(filepath, drive_folder_id=None, share_with_email=DEFAULT_SHARE_EMAIL, retries=3):
    service = get_drive_service()
    filename = os.path.basename(filepath)
    if drive_folder_id is None:
        drive_folder_id = get_folder_id_for_file(filename)
    file_id = find_file_id(filename, drive_folder_id)
    file_metadata = {'name': filename}
    if drive_folder_id:
        file_metadata['parents'] = [drive_folder_id]
    media = MediaFileUpload(filepath, resumable=True)

    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            if file_id:
                updated_file = service.files().update(
                    fileId=file_id,
                    media_body=media
                ).execute()
                print(f"📝 Atualização concluída: {filename} (ID: {file_id})")
                if share_with_email:
                    share_file_with_user(file_id, share_with_email)
                return file_id
            else:
                file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                file_id = file.get('id')
                print(f"✅ Upload concluído: {filename} (ID: {file_id})")
                if share_with_email:
                    share_file_with_user(file_id, share_with_email)
                return file_id
        except Exception as e:
            print(f"⚠️ Tentativa {attempt}/{retries} falhou para upload/update de '{filename}': {e}")
            last_exception = e
            time.sleep(2)
    print(f"❌ Falha definitiva ao enviar '{filename}' após {retries} tentativas: {last_exception}")
    raise last_exception

upload_file = upload_or_update_file

def upload_or_update_all_files_in_directory(local_dir, share_with_email=DEFAULT_SHARE_EMAIL, extensions=('.csv', '.pkl')):
    total = 0
    updated = 0
    uploaded = 0
    failed = 0
    for fname in os.listdir(local_dir):
        if fname.lower().endswith(extensions):
            fpath = os.path.join(local_dir, fname)
            filesize = os.path.getsize(fpath)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[INFO] Processando arquivo: {fname}")
            print(f"       Tamanho: {filesize/1024:.2f} KB")
            print(f"       Modificado em: {mtime}")
            try:
                if fname.lower().endswith('.csv'):
                    drive_folder_id = CSV_FOLDER_ID
                elif fname.lower().endswith('.pkl'):
                    drive_folder_id = PKL_FOLDER_ID
                else:
                    drive_folder_id = None
                file_id = find_file_id(fname, drive_folder_id)
                upload_or_update_file(fpath, drive_folder_id, share_with_email)
                if file_id:
                    updated += 1
                else:
                    uploaded += 1
            except Exception as e:
                print(f"❌ Falha ao processar '{fname}': {e}")
                failed += 1
            total += 1
    print(f"\nResumo do processamento automático:")
    print(f"  Total de arquivos processados: {total}")
    print(f"  Arquivos atualizados (update): {updated}")
    print(f"  Arquivos enviados novos (upload): {uploaded}")
    print(f"  Arquivos com falha definitiva: {failed}")

def download_file(filename, destination_path, drive_folder_id=None, share_with_email=DEFAULT_SHARE_EMAIL):
    try:
        service = get_drive_service()
        file_id = find_file_id(filename, drive_folder_id)
        if not file_id:
            raise FileNotFoundError(f"Arquivo '{filename}' não encontrado no Google Drive na pasta {drive_folder_id}.")
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.close()
        print(f"✅ Download concluído: {filename} (ID: {file_id})")
        if share_with_email:
            share_file_with_user(file_id, share_with_email)
    except Exception as e:
        print(f"❌ Erro ao baixar arquivo {filename}: {e}")
        raise

def list_files_in_drive_folder(drive_folder_id):
    try:
        service = get_drive_service()
        files = []
        page_token = None
        while True:
            response = service.files().list(
                q=f"'{drive_folder_id}' in parents and trashed = false",
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType, modifiedTime, size)',
                pageToken=page_token
            ).execute()
            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
        print(f"\nArquivos na pasta {drive_folder_id}:")
        for f in files:
            print(f"- {f['name']} (ID: {f['id']}, Modificado: {f.get('modifiedTime', '-')}, Tamanho: {f.get('size', '-')} bytes)")
        return files
    except Exception as e:
        print(f"❌ Erro ao listar arquivos na pasta {drive_folder_id}: {e}")
        return []

if __name__ == "__main__":
    local_dir = os.path.join(os.path.dirname(__file__), '.')
    print(f"Iniciando upload/update automático de arquivos do diretório: {local_dir}\n")
    upload_or_update_all_files_in_directory(local_dir)
    print("\nArquivos .csv disponíveis no Google Drive:")
    list_files_in_drive_folder(CSV_FOLDER_ID)
