from data.google_drive_client import upload_or_update_file, DEFAULT_SHARE_EMAIL
import os
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import talib
import tempfile
import time
from pathlib import Path
import uuid
import logging
import joblib

# ==============================================
# CONFIGURAÇÕES CENTRALIZADAS
# ==============================================

@dataclass(frozen=True)
class Config:
    ML_GPT_FOLDER_ID: str = "1Z3uTvxJ-MvBNP1zH7SNR8qI-B__Gg6Lb"  # Pasta para modelos no Gdrive
    DEFAULT_PREFIX: str = "xgb_ml_sgnal"  # Prefixo padrão para arquivos
    MAX_RETRIES: int = 3  # Número máximo de tentativas de upload
    CLEANUP_FILES: bool = True  # Remove arquivos locais após upload
    MAX_FILE_AGE_HOURS: int = 24  # Horas para limpeza de arquivos antigos
    VALID_EXTENSIONS: Tuple[str, ...] = ('.pkl', '.joblib', '.h5', '.onnx')  # Formatos suportados

# Instância global de configuração
config = Config()

# ==============================================
# CONFIGURAÇÃO DE LOGS
# ==============================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('xgb_ml_upload.log')
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# ==============================================
# FUNÇÕES AUXILIARES
# ==============================================
def get_output_dir() -> str:
    """
    Determina o melhor diretório para saída de arquivos em qualquer ambiente.
    Tenta múltiplas localizações até encontrar uma com permissão de escrita.
    """
    candidates = [
        Path(tempfile.gettempdir()) / "xgb_ml_signals",
        Path.cwd() / "output",
        Path.home() / "xgb_ml_temp",
        Path(tempfile.gettempdir()),
    ]
    
    for dir_path in candidates:
        try:
            dir_path.mkdir(exist_ok=True, parents=True, mode=0o755)
            test_file = dir_path / f"test_{uuid.uuid4().hex}.tmp"
            test_file.write_text("test")
            test_file.unlink()
            logger.info(f"Usando diretório de saída: {dir_path}")
            return str(dir_path)
        except (OSError, IOError) as e:
            logger.debug(f"Diretório {dir_path} não acessível: {str(e)}")
            continue
    
    logger.warning("Usando diretório atual como fallback")
    return str(Path.cwd())

def determine_serialization(data: Any) -> Tuple[str, str]:
    """Determina automaticamente o melhor formato de serialização"""
    if hasattr(data, 'predict'):  # Modelo scikit-learn/XGBoost
        return 'joblib', '.joblib'
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return 'pickle', '.pkl'
    elif isinstance(data, dict):
        if any(isinstance(v, (pd.DataFrame, pd.Series)) for v in data.values()):
            return 'pickle', '.pkl'
        return 'joblib', '.joblib'
    return 'pickle', '.pkl'  # Default seguro

def generate_temp_filename(prefix: str = config.DEFAULT_PREFIX, suffix: str = None) -> str:
    """Gera nome de arquivo único com extensão apropriada"""
    output_dir = get_output_dir()
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    suffix = suffix or '.pkl'  # Default para pickle
    filename = f"{prefix}_{timestamp}_{unique_id}{suffix}"
    return str(Path(output_dir) / filename)

# ==============================================
# FUNÇÃO PRINCIPAL DE UPLOAD
# ==============================================
def _perform_upload(filepath: str, folder_id: str, share_with_email: str, max_retries: int) -> Tuple[bool, Optional[str]]:
    """Lógica interna de upload com retry"""
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            file_id = upload_or_update_file(
                filepath=filepath,
                drive_folder_id=folder_id,
                share_with_email=share_with_email
            )
            logger.info(f"Upload concluído | Tentativa: {attempt}/{max_retries}")
            return True, file_id
        except Exception as e:
            last_exception = e
            wait_time = 2 ** attempt
            logger.warning(f"Falha no upload | Tentativa: {attempt}/{max_retries} | Erro: {str(e)}")
            time.sleep(wait_time)
    
    logger.error(f"Falha definitiva após {max_retries} tentativas")
    return False, None

def upload_gpt_ml_model(
    data: Optional[Union[dict, pd.DataFrame, Any]] = None,
    filepath: Optional[str] = None,
    prefix: str = config.DEFAULT_PREFIX,
    share_with_email: str = DEFAULT_SHARE_EMAIL,
    max_retries: int = config.MAX_RETRIES,
    folder_id: str = config.ML_GPT_FOLDER_ID,
    cleanup: bool = config.CLEANUP_FILES
) -> Tuple[bool, Optional[str]]:
    """
    Função principal para upload de modelos/dados para o Google Drive.

    Exemplos de uso:
    1. Upload de dados brutos:
       upload_gpt_ml_model(data={'features': X, 'target': y})
    
    2. Upload de modelo treinado:
       upload_gpt_ml_model(data=modelo_treinado)
    
    3. Upload de arquivo existente:
       upload_gpt_ml_model(filepath='caminho/arquivo.joblib')
    """
    # Validação inicial
    if data is None and filepath is None:
        logger.error("Nenhum dado ou arquivo fornecido")
        return False, None

    temp_file = None
    try:
        # Gerenciamento de arquivo temporário
        if data is not None:
            format, ext = determine_serialization(data)
            temp_file = generate_temp_filename(prefix=prefix, suffix=ext)
            
            try:
                if format == 'joblib':
                    joblib.dump(data, temp_file)
                else:
                    pd.to_pickle(data, temp_file)
                logger.info(f"Dados serializados como {format} em {temp_file}")
                filepath = temp_file
            except Exception as e:
                logger.error(f"Falha na serialização ({format}): {str(e)}")
                return False, None
        else:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        # Validação de extensão
        if not filepath.lower().endswith(config.VALID_EXTENSIONS):
            raise ValueError(f"Formato não suportado: {Path(filepath).suffix}")

        # Processo de upload
        success, file_id = _perform_upload(
            filepath=filepath,
            folder_id=folder_id,
            share_with_email=share_with_email,
            max_retries=max_retries
        )

        # Limpeza pós-upload
        if cleanup and temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"Arquivo temporário removido: {temp_file}")
            except Exception as e:
                logger.warning(f"Falha ao remover arquivo temporário: {str(e)}")

        return success, file_id

    except Exception as e:
        logger.error(f"Erro no processo de upload: {str(e)}", exc_info=True)
        return False, None
    finally:
        # Garante que arquivos temporários sejam removidos em caso de erro
        if temp_file and os.path.exists(temp_file) and not cleanup:
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Falha na limpeza de emergência: {str(e)}")

# ==============================================
# MANUTENÇÃO DE ARQUIVOS TEMPORÁRIOS
# ==============================================
def cleanup_old_files(max_age_hours: int = config.MAX_FILE_AGE_HOURS, prefix: str = config.DEFAULT_PREFIX):
    """Remove arquivos temporários antigos"""
    output_dir = Path(get_output_dir())
    cutoff = time.time() - (max_age_hours * 3600)
    removed_count = 0
    
    for item in output_dir.glob(f"{prefix}_*"):
        try:
            if item.is_file() and item.stat().st_mtime < cutoff:
                item.unlink()
                removed_count += 1
                logger.debug(f"Arquivo antigo removido: {item.name}")
        except Exception as e:
            logger.warning(f"Falha ao remover {item.name}: {str(e)}")
    
    logger.info(f"Limpeza concluída. Arquivos removidos: {removed_count}")

# Limpeza inicial ao carregar o módulo
cleanup_old_files()

# ==============================================
# TESTES (só executaa quando rodar este arquivo diretamente)
# ==============================================
if __name__ == '__main__':
    def run_tests():
        """Testes de unidade - só roda quando executado diretamente"""
        print("\n🔍 Iniciando TESTES do módulo de upload...")
        
        # Teste 1: Upload de dicionário
        print("\nTESTE 1: Upload de dicionário...")
        test_data = {'features': [1, 2, 3], 'target': [0, 1, 0]}
        success, _ = upload_gpt_ml_model(data=test_data, cleanup=False)
        assert success, "❌ Falha no Teste 1"
        print("✅ Teste 1 passou!")
        
        # Teste 2: Upload de DataFrame
        print("\nTESTE 2: Upload de DataFrame...")
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        success, _ = upload_gpt_ml_model(data=df, cleanup=False)
        assert success, "❌ Falha no Teste 2"
        print("✅ Teste 2 passou!")
        
        # Teste 3: Upload de arquivo inválido
        print("\nTESTE 3: Upload de arquivo inválido...")
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                tmp.write(b"dados de teste")
                tmp_path = tmp.name
            success, _ = upload_gpt_ml_model(filepath=tmp_path)
            assert not success, "❌ Teste deveria ter falhado"
            print("✅ Teste 3 passou!")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        print("\n🎉 Todos os testes passaram com sucesso!")

    run_tests()
