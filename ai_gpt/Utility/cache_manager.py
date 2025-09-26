import hashlib
import json
from typing import Any, Dict, Optional, Callable
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from collections import OrderedDict
import sys
import os

# -------------------------------
# Helpers de serialização / hash
# -------------------------------
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _serialize_param(obj: Any, _seen: set = None) -> Any:
    """
    Canonicaliza objetos Python em estruturas pequenas e determinísticas.
    Evita serializar arrays enormes (usa amostras para hash).
    Suporta: None, bool, int, float, str, datetime, timedelta, numpy, pandas, dict, list, set, tuple.
    """
    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return "<recursion>"
    _seen.add(obj_id)

    # Primitivos
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return format(obj, ".12g")
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()

    # numpy scalar
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return float(obj)

    # numpy array (amostragem otimizada quando grande)
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        flat = arr.ravel()
        
        # Para arrays muito grandes, usar estatísticas sumárias em vez de amostras
        if flat.nbytes > 1024 * 1024:  # >1MB -> usar estatísticas
            try:
                stats = {
                    'mean': float(np.mean(flat)),
                    'std': float(np.std(flat)),
                    'min': float(np.min(flat)),
                    'max': float(np.max(flat)),
                    'size': flat.size,
                    'dtype': str(arr.dtype),
                    'shape': arr.shape
                }
                # Adicionar hash das primeiras e últimas amostras para detecção de mudanças
                sample_size = min(100, flat.size)
                if sample_size > 0:
                    sample = np.concatenate([flat[:sample_size//2], flat[-sample_size//2:]])
                    stats['sample_hash'] = _hash_bytes(np.ascontiguousarray(sample).tobytes())
                return {"__ndarray_stats__": stats}
            except Exception:
                # Fallback para amostragem simples se cálculo de stats falhar
                sample_size = min(2000, flat.size)
                sample = flat[:sample_size] if sample_size == flat.size else np.concatenate([
                    flat[:sample_size//2], flat[-sample_size//2:]
                ])
                sample_bytes = np.ascontiguousarray(sample).tobytes()
                return {"__ndarray__": True, "shape": arr.shape, "dtype": str(arr.dtype),
                        "hash": _hash_bytes(sample_bytes), "sample_size": sample_size}
        else:
            # Para arrays menores, manter abordagem original
            sample_bytes = arr.tobytes()
            return {"__ndarray__": True, "shape": arr.shape, "dtype": str(arr.dtype),
                    "hash": _hash_bytes(sample_bytes)}

    # pandas Series - otimizado para séries grandes
    if isinstance(obj, pd.Series):
        vals = obj.values
        if vals.nbytes > 1024 * 1024:  # >1MB
            try:
                stats = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'size': vals.size,
                    'dtype': str(obj.dtype)
                }
                sample_size = min(100, vals.size)
                if sample_size > 0:
                    sample = np.concatenate([vals[:sample_size//2], vals[-sample_size//2:]])
                    stats['sample_hash'] = _hash_bytes(np.ascontiguousarray(sample).tobytes())
                idx_sample = list(obj.index[:5]) + list(obj.index[-5:]) if len(obj.index) > 10 else list(obj.index)
                return {"__series_stats__": stats, "index_sample": idx_sample}
            except Exception:
                # Fallback para abordagem original se stats falharem
                vals_hash = _hash_bytes(np.ascontiguousarray(vals).tobytes()) if vals.size else _hash_bytes(b"")
                idx_hash = _hash_bytes(json.dumps(list(map(str, obj.index)), sort_keys=True).encode())
                return {"__series__": True, "dtype": str(obj.dtype), "shape": obj.shape,
                        "index_hash": idx_hash, "hash": vals_hash}
        else:
            # Mantém abordagem original para séries menores
            vals_hash = _hash_bytes(np.ascontiguousarray(vals).tobytes()) if vals.size else _hash_bytes(b"")
            idx_hash = _hash_bytes(json.dumps(list(map(str, obj.index)), sort_keys=True).encode())
            return {"__series__": True, "dtype": str(obj.dtype), "shape": obj.shape,
                    "index_hash": idx_hash, "hash": vals_hash}

    # pandas DataFrame - otimizado para DataFrames grandes
    if isinstance(obj, pd.DataFrame):
        vals = obj.values
        if vals.nbytes > 1024 * 1024:  # >1MB
            try:
                stats = {
                    'shape': obj.shape,
                    'columns': list(obj.columns),
                    'dtypes': {col: str(dtype) for col, dtype in obj.dtypes.items()}
                }
                # Estatísticas para as primeiras 5 colunas numéricas
                numeric_cols = obj.select_dtypes(include=[np.number]).columns.tolist()[:5]
                for col in numeric_cols:
                    if col in obj.columns:
                        stats[f'{col}_mean'] = float(obj[col].mean())
                        stats[f'{col}_std'] = float(obj[col].std())
                
                # Hash de amostra para detecção de mudanças
                sample_size = min(100, len(obj))
                if sample_size > 0:
                    sample = obj.iloc[np.concatenate([range(0, sample_size//2), range(-sample_size//2, 0)])]
                    sample_hash = _hash_bytes(sample.to_csv(index=False).encode())
                    stats['sample_hash'] = sample_hash
                return {"__dataframe_stats__": stats}
            except Exception:
                # Fallback para abordagem original se stats falharem
                vals_hash = _hash_bytes(np.ascontiguousarray(vals).tobytes()) if vals.size else _hash_bytes(b"")
                cols_hash = _hash_bytes(json.dumps(list(map(str, obj.columns)), sort_keys=True).encode())
                idx_hash = _hash_bytes(json.dumps(list(map(str, obj.index)), sort_keys=True).encode())
                return {"__dataframe__": True, "shape": obj.shape,
                        "cols_hash": cols_hash, "index_hash": idx_hash, "hash": vals_hash}
        else:
            # Mantém abordagem original para DataFrames menores
            vals_hash = _hash_bytes(np.ascontiguousarray(vals).tobytes()) if vals.size else _hash_bytes(b"")
            cols_hash = _hash_bytes(json.dumps(list(map(str, obj.columns)), sort_keys=True).encode())
            idx_hash = _hash_bytes(json.dumps(list(map(str, obj.index)), sort_keys=True).encode())
            return {"__dataframe__": True, "shape": obj.shape,
                    "cols_hash": cols_hash, "index_hash": idx_hash, "hash": vals_hash}

    # dict -> ordenar chaves para determinismo
    if isinstance(obj, dict):
        return {str(k): _serialize_param(obj[k], _seen) for k in sorted(obj.keys(), key=str)}

    # listas / tuplas / sets
    if isinstance(obj, (list, tuple, set)):
        return [_serialize_param(x, _seen) for x in list(obj)]

    # fallback seguro
    try:
        return str(obj)
    except Exception:
        return repr(obj)

def make_cache_key(pair: str, operation_type: str, params: Dict[str, Any],
                   timeframes: Optional[list] = None,
                   cache_namespace: str = "sr_analyzer", cache_version: int = 1,
                   hash_len: int = 12) -> str:
    """
    Gera uma chave consistente baseada em:
      - pair
      - operation_type
      - múltiplos timeframes (ex: base_tf, aux_tf, e futuramente signal_tf)
      - params canonificados
    """
    pair_s = str(pair).strip()
    op_s = str(operation_type).strip()
    tfs = "_".join([str(tf).strip() for tf in timeframes]) if timeframes else "default"

    try:
        canon = _serialize_param(params)
        payload = {
            "pair": pair_s,
            "op": op_s,
            "tfs": tfs,  # <- base_tf + aux_tf (futuramente signal_tf pode ser adicionado aqui)
            "params": canon,
            "ver": int(cache_version)
        }
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":"), default=str)
    except Exception as e:
        payload_json = json.dumps({"fallback": f"{pair_s}_{op_s}_{tfs}_{str(e)[:32]}"}, sort_keys=True)

    key_hash = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()[:hash_len]
    safe_pair = pair_s.replace(" ", "_").replace("/", "_")
    safe_op = op_s.replace(" ", "_")
    safe_tfs = tfs.replace(" ", "_")
    return f"{cache_namespace}:{safe_pair}_{safe_op}_{safe_tfs}_{key_hash}"

# -------------------------------
# CacheManager com invalidação por preço e limite de memória
# -------------------------------
class CacheManager:
    def __init__(self, default_expiry: int = 600, max_size_mb: int = 100):
        """
        default_expiry: segundos (padrão 600 = 10 minutos)
        max_size_mb: tamanho máximo em MB que o cache pode ocupar (padrão 100MB)
        Cada entrada guarda:
          - value: qualquer objeto serializável em memória
          - timestamp: datetime do set
          - expiry: segundos de validade
          - meta: dict opcional (ex: last_price, price_tolerance, tags)
          - size: tamanho aproximado em bytes da entrada
        """
        self._store = OrderedDict()
        self.default_expiry = default_expiry
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0

    def _estimate_size(self, value: Any) -> int:
        """Estima o tamanho em bytes de um objeto para controle de memória"""
        try:
            if hasattr(value, '__sizeof__'):
                return sys.getsizeof(value)
            elif isinstance(value, (str, bytes, bytearray)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 24  # Tamanho aproximado de números em Python
            elif isinstance(value, (list, tuple, set)):
                return sum(self._estimate_size(item) for item in value) + 50
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items()) + 100
            else:
                # Para objetos complexos, usar estimativa conservadora
                return 1024
        except Exception:
            # Em caso de erro, retornar tamanho padrão
            return 1024

    def _evict_if_needed(self):
        """Remove entradas mais antidas se o cache exceder o tamanho máximo"""
        while self.current_size > self.max_size_bytes and self._store:
            # Remove a entrada mais antiga (primeira no OrderedDict)
            key, entry = self._store.popitem(last=False)
            self.current_size -= entry.get("size", 0)

    def set(self, key: str, value: Any, *, expiry: Optional[int] = None,
            last_price: Optional[float] = None, price_tolerance: Optional[float] = None,
            tags: Optional[Dict[str, Any]] = None):
        """
        Armazena um item no cache.
        - key: chave (gerada por make_cache_key ou manual)
        - value: payload (ex: levels)
        - expiry: segundos (opcional)
        - last_price: preço no momento do cálculo (para invalidação baseada em price)
        - price_tolerance: tolerância relativa (fractional, ex: 0.005 = 0.5%). Se passado, get_with_price usará isto.
        - tags: dicionário livre para metadados.
        """
        # Estimar tamanho da entrada
        entry_size = self._estimate_size(value)
        entry = {
            "value": value,
            "timestamp": datetime.now(),
            "expiry": expiry or self.default_expiry,
            "size": entry_size,
            "meta": {
                "last_price": float(last_price) if last_price is not None else None,
                "price_tolerance": float(price_tolerance) if price_tolerance is not None else None,
                "tags": tags or {}
            }
        }
        
        # Se a chave já existe, subtrair seu tamanho atual
        if key in self._store:
            self.current_size -= self._store[key].get("size", 0)
        
        # Adicionar ao cache e atualizar tamanho
        self._store[key] = entry
        self.current_size += entry_size
        
        # Remover entradas se necessário para liberar espaço
        self._evict_if_needed()

    def get(self, key: str) -> Optional[Any]:
        """
        Retorna o valor se existir e não estiver expirado. Não faz checagem de preço.
        Use get_with_price() se quiser invalidação por preço.
        """
        entry = self._store.get(key)
        if not entry:
            return None
            
        # Mover para o final (LRU - indicando uso recente)
        self._store.move_to_end(key)
        ts = entry["timestamp"]
        exp = timedelta(seconds=entry["expiry"])
        if datetime.now() - ts > exp:
            self.current_size -= entry.get("size", 0)
            self._store.pop(key, None)
            return None
        return entry["value"]

    def get_with_price(self, key: str, current_price: Optional[float] = None,
                       *, price_tolerance: Optional[float] = None) -> Optional[Any]:
        """
        Retorna o valor se não expirado e se a variação de preço não exceder tolerância.
        Lógica de decisão:
          1) checa expiry temporário
          2) obtém last_price e price_tolerance salvos em meta (se houver)
          3) usa price_tolerance param se meta não definir
          4) se current_price for None, não faz validação por preço (apenas expiry)
        """
        entry = self._store.get(key)
        if not entry:
            return None
            
        # Mover para o final (LRU - indicando uso recente)
        self._store.move_to_end(key)

        # 1) expiry
        ts = entry["timestamp"]
        exp = timedelta(seconds=entry["expiry"])
        if datetime.now() - ts > exp:
            self.current_size -= entry.get("size", 0)
            self._store.pop(key, None)
            return None

        # 2) validação por preço
        meta = entry.get("meta", {})
        last_price = meta.get("last_price")
        saved_tol = meta.get("price_tolerance")

        # resolução da tolerância (prioridade: argumento > meta)
        tol = price_tolerance if price_tolerance is not None else saved_tol
        if current_price is not None and last_price is not None and tol is not None:
            try:
                rel_change = abs(float(current_price) - float(last_price)) / max(abs(float(last_price)), 1e-9)
                if rel_change > float(tol):
                    self.current_size -= entry.get("size", 0)
                    self._store.pop(key, None)
                    return None
            except Exception:
                # em caso de qualquer erro, optamos por invalidar (mais seguro)
                self.current_size -= entry.get("size", 0)
                self._store.pop(key, None)
                return None
        return entry["value"]

    def get_or_refresh(self, key: str, refresh_fn: Callable[[], Any],
                       *, current_price: Optional[float] = None,
                       expiry: Optional[int] = None,
                       last_price: Optional[float] = None,
                       price_tolerance: Optional[float] = None,
                       tags: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Recupera um item do cache.
        - Se o item expirou ou a variação de preço excedeu a tolerância:
            -> chama refresh_fn() para recalcular
            -> atualiza o cache automaticamente
        - Caso contrário, retorna o valor atual do cache.
        """
        val = self.get_with_price(key, current_price=current_price, price_tolerance=price_tolerance)
        if val is not None:
            return val

        # Se o cache não é válido, recalcular
        try:
            new_val = refresh_fn()
            if new_val is not None:
                self.set(
                    key, new_val,
                    expiry=expiry,
                    last_price=last_price if last_price is not None else current_price,
                    price_tolerance=price_tolerance,
                    tags=tags
                )
            return new_val
        except Exception as e:
            print(f"[CacheManager] Erro no auto-refresh para {key}: {e}")
            return None

    def invalidate(self, key: str):
        """Remove uma entrada específica do cache"""
        if key in self._store:
            self.current_size -= self._store[key].get("size", 0)
            self._store.pop(key, None)

    def invalidate_by_predicate(self, predicate: Callable[[str, Dict[str, Any]], bool]):
        """
        Remove entradas para as quais predicate(key, entry) retorna True.
        Útil para invalidar por tags, par, timeframe, etc.
        """
        keys_to_remove = [k for k, v in self._store.items() if predicate(k, v)]
        for k in keys_to_remove:
            self.current_size -= self._store[k].get("size", 0)
            self._store.pop(k, None)

    def clear(self):
        """Limpa todo o cache"""
        self._store.clear()
        self.current_size = 0

    def stats(self) -> Dict[str, Any]:
        """
        Estatísticas básicas: número de itens, chaves amostradas, uso de memória.
        """
        keys = list(self._store.keys())
        return {
            "items": len(keys),
            "memory_usage_mb": self.current_size / (1024 * 1024),
            "memory_limit_mb": self.max_size_bytes / (1024 * 1024),
            "sample_keys": keys[:5]  # Mostrar apenas 5 chaves para não poluir
        }

    def set_for(self, pair: str, operation_type: str, params: Dict[str, Any],
                value: Any, *, timeframes: Optional[list] = None,
                expiry: Optional[int] = None, last_price: Optional[float] = None,
                price_tolerance: Optional[float] = None, tags: Optional[Dict[str, Any]] = None,
                cache_namespace: str = "sr_analyzer", cache_version: int = 1,
                hash_len: int = 12) -> str:
        key = make_cache_key(pair, operation_type, params, timeframes=timeframes,
                             cache_namespace=cache_namespace, cache_version=cache_version, hash_len=hash_len)
        self.set(key, value, expiry=expiry, last_price=last_price,
                 price_tolerance=price_tolerance, tags=tags)
        return key
  
    def get_for(self, pair: str, operation_type: str, params: Dict[str, Any],
                current_price: Optional[float] = None, *, timeframes: Optional[list] = None,
                price_tolerance: Optional[float] = None,
                cache_namespace: str = "sr_analyzer", cache_version: int = 1,
                hash_len: int = 12) -> Optional[Any]:
        """
        Recupera usando chave gerada (pair + operation_type + timeframes + params).
        Usa get_with_price() se current_price for passado.
        """
        key = make_cache_key(pair, operation_type, params, timeframes=timeframes,
                             cache_namespace=cache_namespace,
                             cache_version=cache_version,
                             hash_len=hash_len)
        if current_price is None:
            return self.get(key)
        return self.get_with_price(key, current_price, price_tolerance=price_tolerance)

    def get_for_or_refresh(self, pair: str, operation_type: str, params: Dict[str, Any],
                           refresh_fn: Callable[[], Any],
                           current_price: Optional[float] = None, *,
                           timeframes: Optional[list] = None,
                           expiry: Optional[int] = None,
                           last_price: Optional[float] = None,
                           price_tolerance: Optional[float] = None,
                           tags: Optional[Dict[str, Any]] = None,
                           cache_namespace: str = "sr_analyzer", cache_version: int = 1,
                           hash_len: int = 12) -> Optional[Any]:
        """
        Versão conveniente de get_or_refresh que gera a chave a partir de pair+operation+timeframes+params.
        """
        key = make_cache_key(pair, operation_type, params, timeframes=timeframes,
                             cache_namespace=cache_namespace,
                             cache_version=cache_version,
                             hash_len=hash_len)
        return self.get_or_refresh(
            key, refresh_fn,
            current_price=current_price,
            expiry=expiry,
            last_price=last_price,
            price_tolerance=price_tolerance,
            tags=tags
        )

    # Persistência em disco ------------------------
    def save_to_disk(self, folder: str):
        """Salva o cache inteiro em disco (JSON + Parquet para DataFrames grandes)."""
        try:
            os.makedirs(folder, exist_ok=True)
            meta = {}
            for k, entry in self._store.items():
                val = entry["value"]
                if isinstance(val, pd.DataFrame):
                    path = os.path.join(folder, f"{k.replace(':','_')}.parquet")
                    val.to_parquet(path, index=True)
                    meta[k] = {**entry, "value": {"__dataframe__": path}}
                else:
                    meta[k] = entry
            with open(os.path.join(folder, "cache_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, default=str)
            return True
        except Exception as e:
            print(f"[CacheManager] Erro ao salvar cache: {e}")
            return False

    def load_from_disk(self, folder: str, restore_expired: bool = False):
        """Carrega cache de disco. Se restore_expired=False, ignora entradas vencidas."""
        try:
            meta_path = os.path.join(folder, "cache_meta.json")
            if not os.path.exists(meta_path):
                return False
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            now = datetime.now()
            for k, entry in meta.items():
                ts = entry.get("timestamp")
                exp = timedelta(seconds=entry.get("expiry", self.default_expiry))
                if restore_expired or (ts and (now - datetime.fromisoformat(str(ts)) < exp)):
                    val = entry["value"]
                    if isinstance(val, dict) and "__dataframe__" in val:
                        path = val["__dataframe__"]
                        if os.path.exists(path):
                            entry["value"] = pd.read_parquet(path)
                    self._store[k] = entry
                    self.current_size += entry.get("size", 0)
            self._evict_if_needed()
            return True
        except Exception as e:
            print(f"[CacheManager] Erro ao carregar cache: {e}")
            return False

# Instância global (para importar facilmente)
global_cache = CacheManager()
