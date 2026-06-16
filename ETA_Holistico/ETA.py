"""
Motor de integración: Predicción de tráfico + NEXTPARK
=======================================================
Integra el modelo LSTM de predicción de tráfico (Laura) con el algoritmo
NEXTPARK de predicción de aparcamiento para calcular un ETA holístico:
 
    ETA_total = T_conducción + T_maniobra_aparcamiento + T_caminata_a_pie
 
Autores: Next Mobility Solutions
Proyecto: PERTE VEC — Expediente VE2-010000-2023-75
"""
 
from __future__ import annotations
 
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
 
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import MultiPoint
 
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NextParkIntegration")
 
 
# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------
 
@dataclass
class TrafficPrediction:
    """Resultado del modelo LSTM de predicción de tráfico."""
    node_id: str
    travel_time_min: float          # Tiempo de conducción estimado (minutos)
    predicted_intensity: float      # Intensidad predicha (veh/hora)
    predicted_speed_kmh: float      # Velocidad media predicha (km/h)
    timestamp: datetime = field(default_factory=datetime.now)
 
 
@dataclass
class ParkingSignal:
    """Señal de intención de aparcamiento generada por NEXTPARK."""
    aparca_en_10min: bool           # El vehículo aparcará en los próximos 10 min
    time_to_park_seg: float         # Tiempo estimado hasta aparcar (segundos)
    zona_frecuente: bool            # El conductor está en una zona habitual
    velocidad_ventana_kmh: float    # Velocidad media en los últimos 5 min
    confianza: str                  # "alta" | "media" | "baja"
 
 
@dataclass
class ETAHolistico:
    """ETA total desglosado en sus tres componentes."""
    parking_id: str
    parking_nombre: str
    parking_coords: tuple[float, float]
    parking_ocupacion: float
 
    # Componentes del ETA (en minutos)
    t_conduccion_min: float
    t_maniobra_min: float
    t_caminata_min: float
    eta_total_min: float
 
    # Desglose adicional
    distancia_caminata_m: float
    velocidad_caminata_kmh: float = 4.8
 
    def resumen(self) -> str:
        return (
            f"[{self.parking_nombre}] ETA total: {self.eta_total_min:.1f} min "
            f"(conducción: {self.t_conduccion_min:.1f} | "
            f"maniobra: {self.t_maniobra_min:.1f} | "
            f"caminata: {self.t_caminata_min:.1f})"
        )
 
 
# ---------------------------------------------------------------------------
# Módulo 1: Predicción de tráfico (interfaz con el modelo LSTM de Laura)
# ---------------------------------------------------------------------------
 
class TrafficPredictor:
    """
    Interfaz con el modelo LSTM de predicción de tráfico.
 
    En producción, este módulo carga el modelo LSTM entrenado (Keras/TFLite)
    y ejecuta la inferencia sobre los datos en tiempo real del Ayuntamiento
    de Madrid. Aquí se expone la interfaz pública que el motor de integración
    consume, independientemente de la implementación interna del modelo.
 
    Parámetros del modelo (arquitectura seleccionada en e06.1):
        - 2 capas LSTM (128 + 64 neuronas)
        - Variables cíclicas (seno/coseno de hora, día, mes)
        - Variables de entrada: intensidad, ocupación, carga, vmed + cíclicas
        - Salidas: intensidad predicha, velocidad media predicha
    """
 
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Ruta al modelo LSTM entrenado (.h5 o SavedModel).
                        Si es None, se usa el modo simulación para desarrollo.
        """
        self.model_path = model_path
        self.model = None
        self._simulation_mode = model_path is None
 
        if not self._simulation_mode:
            self._cargar_modelo()
        else:
            logger.warning(
                "TrafficPredictor en modo simulación. "
                "Proporciona model_path para usar el modelo real."
            )
 
    def _cargar_modelo(self):
        """Carga el modelo LSTM desde disco."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Modelo LSTM cargado desde {self.model_path}")
        except ImportError:
            raise RuntimeError(
                "TensorFlow no está instalado. "
                "Ejecuta: pip install tensorflow"
            )
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo: {e}")
 
    def _preparar_features(
        self,
        historial_nodo: pd.DataFrame,
        timestamp: datetime,
    ) -> np.ndarray:
        """
        Prepara el vector de features para el modelo LSTM.
 
        El modelo fue entrenado con las siguientes variables:
            intensidad, ocupacion, carga, vmed,
            hora_sin, hora_cos, dia_sin, dia_cos, mes_sin, mes_cos,
            tipo_elemento (urbano=1, interurbano=0), error_flag
 
        Args:
            historial_nodo: DataFrame con el historial del nodo (últimas N lecturas).
            timestamp: Momento de la predicción.
 
        Returns:
            Array de shape (1, seq_len, n_features) listo para el modelo.
        """
        df = historial_nodo.copy()
 
        # Variables cíclicas temporales
        hora = timestamp.hour + timestamp.minute / 60
        dia = timestamp.weekday()
        mes = timestamp.month
 
        df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
        df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
        df["dia_sin"] = np.sin(2 * np.pi * dia / 7)
        df["dia_cos"] = np.cos(2 * np.pi * dia / 7)
        df["mes_sin"] = np.sin(2 * np.pi * mes / 12)
        df["mes_cos"] = np.cos(2 * np.pi * mes / 12)
 
        feature_cols = [
            "intensidad", "ocupacion", "carga", "vmed",
            "hora_sin", "hora_cos", "dia_sin", "dia_cos",
            "mes_sin", "mes_cos", "tipo_elemento", "error_flag",
        ]
 
        # Verificar que existen todas las columnas necesarias
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el historial del nodo: {missing}")
 
        X = df[feature_cols].values
        return X.reshape(1, X.shape[0], X.shape[1])
 
    def predecir(
        self,
        node_id: str,
        historial_nodo: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
    ) -> TrafficPrediction:
        """
        Genera una predicción de tráfico para un nodo dado.
 
        Args:
            node_id: Identificador del punto de medida (ej: "6798").
            historial_nodo: DataFrame con el historial reciente del nodo.
                            Columnas requeridas: intensidad, ocupacion, carga,
                            vmed, tipo_elemento, error_flag.
            timestamp: Momento para el que se genera la predicción.
                       Si es None, se usa el momento actual.
 
        Returns:
            TrafficPrediction con el tiempo de conducción estimado.
        """
        if timestamp is None:
            timestamp = datetime.now()
 
        if self._simulation_mode:
            return self._predecir_simulacion(node_id, timestamp)
 
        X = self._preparar_features(historial_nodo, timestamp)
        prediccion = self.model.predict(X, verbose=0)[0]
 
        intensidad_pred = float(prediccion[0])
        velocidad_pred = float(prediccion[1])
 
        # Tiempo de conducción estimado a partir de la velocidad predicha
        # (el grafo de rutas calculará el tiempo real tramo a tramo)
        travel_time = self._estimar_tiempo_conduccion(velocidad_pred)
 
        return TrafficPrediction(
            node_id=node_id,
            travel_time_min=travel_time,
            predicted_intensity=intensidad_pred,
            predicted_speed_kmh=velocidad_pred,
            timestamp=timestamp,
        )
 
    def _estimar_tiempo_conduccion(self, velocidad_kmh: float) -> float:
        """
        Estima el tiempo de conducción en minutos.
        En producción, esto se calcula tramo a tramo con el grafo de Dijkstra.
        """
        # Distancia media al área de destino (km) — parametrizable
        distancia_media_km = 5.0
        velocidad_efectiva = max(velocidad_kmh, 5.0)  # Mínimo 5 km/h
        return (distancia_media_km / velocidad_efectiva) * 60
 
    def _predecir_simulacion(
        self, node_id: str, timestamp: datetime
    ) -> TrafficPrediction:
        """Modo simulación para desarrollo sin modelo cargado."""
        hora = timestamp.hour
        # Simular picos de tráfico en hora punta (7-9h y 18-20h)
        es_hora_punta = (7 <= hora <= 9) or (18 <= hora <= 20)
        intensidad = 2200.0 if es_hora_punta else 800.0
        velocidad = 15.0 if es_hora_punta else 40.0
        travel_time = self._estimar_tiempo_conduccion(velocidad)
 
        logger.debug(
            f"[SIMULACIÓN] Nodo {node_id}: intensidad={intensidad}, "
            f"velocidad={velocidad} km/h, tiempo={travel_time:.1f} min"
        )
        return TrafficPrediction(
            node_id=node_id,
            travel_time_min=travel_time,
            predicted_intensity=intensidad,
            predicted_speed_kmh=velocidad,
            timestamp=timestamp,
        )
 
 
# ---------------------------------------------------------------------------
# Módulo 2: Señal de aparcamiento NEXTPARK
# ---------------------------------------------------------------------------
 
class NextParkPredictor:
    """
    Implementación en Python del algoritmo NEXTPARK.
 
    Basado en los notebooks NEXTPARK-e03 a NEXTPARK-e07.
    Detecta la intención de aparcamiento a partir del comportamiento
    cinemático del vehículo en los últimos minutos.
 
    Modelo: Regresión lineal con transformación logarítmica de la variable
    respuesta (time_to_park), entrenado sobre ~41.000 viajes reales.
 
    Coeficientes del modelo definitivo (notebook e06.1):
        log(time_to_park + 1) ~ area_cubierta + var_velocidad +
                                 amplitud_menor + distancia_cubierta +
                                 velocidad_ventana
    """
 
    # Coeficientes del modelo definitivo (extraídos del notebook e06.1)
    _COEF_INTERCEPT    =  6.025781624
    _COEF_AREA         = -0.012842753
    _COEF_VAR_VEL      = -0.004247631
    _COEF_AMPLITUD     = -0.003427578
    _COEF_DISTANCIA    =  0.100941208
    _COEF_VEL_VENTANA  = -0.006983576
 
    # Límites de truncado (del notebook e05 — filtrado de anomalías)
    _MAX_VELOCIDAD_VENTANA  = 180.0   # km/h
    _MAX_VAR_VELOCIDAD      = 100.0   # km/h
    _MAX_AMPLITUD           = 360.0   # grados
    _MAX_DISTANCIA          = 30.0    # km
    _MAX_AREA               = 80.0    # km²
    _MAX_TIME_TO_PARK       = 1200.0  # segundos (20 min máximo)
 
    # Ventanas temporales del modelo (optimizadas en e03)
    _VENTANA_AREA_MIN        = 20
    _VENTANA_VAR_VEL_MIN     = 10
    _VENTANA_AMPLITUD_MIN    = 5
    _VENTANA_DISTANCIA_MIN   = 5
 
    # Umbral de velocidad para emisión de alarma (del análisis e06.1)
    _UMBRAL_VELOCIDAD_ALARMA = 40.0   # km/h
 
    # Radio de detección de zonas frecuentes (notebook e02)
    _RADIO_ZONA_FRECUENTE_M  = 250.0
    _DISTANCIA_GRADOS        = 250.0 / (111320 * math.cos(40 * math.pi / 180))
 
    def __init__(self, pf_folder: str = "./pf"):
        """
        Args:
            pf_folder: Carpeta con los CSV de puntos frecuentes por conductor.
                       Formato esperado: {conductor_id}_pf.csv
        """
        self.pf_folder = pf_folder
        self._cache_pf: dict[str, pd.DataFrame] = {}
 
    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
 
    def generar_senal(
        self,
        historial_viaje: pd.DataFrame,
        conductor_id: str,
        timestamp: Optional[datetime] = None,
    ) -> ParkingSignal:
        """
        Genera una señal de intención de aparcamiento.
 
        Args:
            historial_viaje: DataFrame con el historial del viaje actual.
                             Columnas requeridas: Latitud, Longitud, FechaRTC.
                             Debe estar ordenado por FechaRTC ascendente.
            conductor_id: Identificador del conductor (para cargar sus zonas frecuentes).
            timestamp: Momento de la predicción. Si es None, usa el último registro.
 
        Returns:
            ParkingSignal con la predicción y metadatos de confianza.
        """
        df = historial_viaje.copy()
        df["FechaRTC"] = pd.to_datetime(df["FechaRTC"])
        df = df.sort_values("FechaRTC", ascending=True).reset_index(drop=True)
 
        if timestamp is None:
            timestamp = df.iloc[-1]["FechaRTC"]
 
        # Extraer ventanas temporales
        datos_20min = self._ventana(df, timestamp, self._VENTANA_AREA_MIN)
        datos_10min = self._ventana(df, timestamp, self._VENTANA_VAR_VEL_MIN)
        datos_5min  = self._ventana(df, timestamp, self._VENTANA_AMPLITUD_MIN)
 
        # Calcular features
        area_cubierta       = self._calcular_area(datos_20min)
        var_velocidad       = self._calcular_var_velocidad(datos_10min)
        amplitud_menor      = self._calcular_amplitud(datos_5min)
        distancia_cubierta, vel_ventana = self._calcular_distancia_velocidad(datos_5min)
 
        # Truncar valores extremos (consistente con el entrenamiento)
        area_cubierta    = min(area_cubierta, self._MAX_AREA)
        var_velocidad    = min(var_velocidad, self._MAX_VAR_VELOCIDAD)
        amplitud_menor   = min(amplitud_menor, self._MAX_AMPLITUD)
        distancia_cubierta = min(distancia_cubierta, self._MAX_DISTANCIA)
        vel_ventana      = min(vel_ventana, self._MAX_VELOCIDAD_VENTANA)
 
        # Predicción del modelo
        time_to_park_seg = self._predecir_time_to_park(
            area_cubierta, var_velocidad, amplitud_menor,
            distancia_cubierta, vel_ventana,
        )
        time_to_park_seg = min(time_to_park_seg, self._MAX_TIME_TO_PARK)
 
        # Detección de zona frecuente
        lat_actual = df.iloc[-1]["Latitud"]
        lon_actual = df.iloc[-1]["Longitud"]
        zona_frec = self._es_zona_frecuente(conductor_id, lat_actual, lon_actual)
 
        # Lógica de alarma (umbral 10 min + velocidad baja + no zona frecuente)
        aparca_en_10min = (
            time_to_park_seg <= 600
            and vel_ventana <= self._UMBRAL_VELOCIDAD_ALARMA
            and not zona_frec
            and area_cubierta > 0  # Filtro anomalías
        )
 
        confianza = self._evaluar_confianza(
            vel_ventana, zona_frec, area_cubierta, var_velocidad
        )
 
        logger.debug(
            f"[NEXTPARK] Conductor {conductor_id}: "
            f"time_to_park={time_to_park_seg:.0f}s, "
            f"vel={vel_ventana:.1f} km/h, "
            f"zona_frec={zona_frec}, "
            f"alarma={aparca_en_10min}, "
            f"confianza={confianza}"
        )
 
        return ParkingSignal(
            aparca_en_10min=aparca_en_10min,
            time_to_park_seg=time_to_park_seg,
            zona_frecuente=zona_frec,
            velocidad_ventana_kmh=vel_ventana,
            confianza=confianza,
        )
 
    # ------------------------------------------------------------------
    # Features del modelo
    # ------------------------------------------------------------------
 
    def _ventana(
        self, df: pd.DataFrame, t_ref: datetime, minutos: int
    ) -> pd.DataFrame:
        """Filtra el DataFrame a los últimos N minutos respecto a t_ref."""
        corte = pd.Timestamp(t_ref) - pd.Timedelta(minutes=minutos)
        subset = df[df["FechaRTC"] >= corte]
        return subset if len(subset) >= 2 else df.tail(2)
 
    def _calcular_area(self, df: pd.DataFrame) -> float:
        """Área del convex hull de los puntos de los últimos 20 min (km²)."""
        try:
            if len(df) < 3:
                return 0.0
            puntos = MultiPoint(df[["Longitud", "Latitud"]].values)
            hull = puntos.convex_hull
            if not hull.is_valid or hull.geom_type == "Point":
                return 0.0
            return float(hull.area)
        except Exception as e:
            logger.warning(f"Error al calcular área cubierta: {e}")
            return 0.0
 
    def _calcular_var_velocidad(self, df: pd.DataFrame) -> float:
        """Desviación estándar de la velocidad en los últimos 10 min (km/h)."""
        try:
            df = df.copy()
            df["FechaRTC"] = pd.to_datetime(df["FechaRTC"])
            tiempo_seg = df["FechaRTC"].diff().dt.total_seconds()
 
            distancias = [
                geodesic((lat1, lon1), (lat2, lon2)).meters / 1000
                if not (pd.isna(lat1) or pd.isna(lon1))
                else 0.0
                for lat1, lon1, lat2, lon2 in zip(
                    df["Latitud"].shift(), df["Longitud"].shift(),
                    df["Latitud"], df["Longitud"],
                )
            ]
 
            velocidades = np.array(distancias) / (tiempo_seg.fillna(1) / 3600)
            velocidades = np.where(np.isfinite(velocidades), velocidades, 0.0)
            return float(np.std(velocidades))
        except Exception as e:
            logger.warning(f"Error al calcular variación de velocidad: {e}")
            return 0.0
 
    def _calcular_amplitud(self, df: pd.DataFrame) -> float:
        """Amplitud mínima del ángulo de giro en los últimos 5 min (grados)."""
        try:
            if len(df) < 2:
                return 0.0
            angulo = np.round(
                np.arctan2(
                    df["Longitud"].values - np.roll(df["Longitud"].values, 1),
                    df["Latitud"].values - np.roll(df["Latitud"].values, 1),
                ) * (180 / math.pi),
                1,
            )
            angulo[0] = angulo[1] if len(angulo) > 1 else 0.0
            angulo_360 = np.where(angulo < 0, angulo + 360, angulo)
 
            amp_normal = abs(float(angulo.max()) - float(angulo.min()))
            amp_360    = abs(float(angulo_360.max()) - float(angulo_360.min()))
            return min(amp_normal, amp_360)
        except Exception as e:
            logger.warning(f"Error al calcular amplitud de ángulo: {e}")
            return 0.0
 
    def _calcular_distancia_velocidad(
        self, df: pd.DataFrame
    ) -> tuple[float, float]:
        """
        Distancia recorrida (km) y velocidad media (km/h) en los últimos 5 min.
        """
        try:
            distancias = [
                geodesic((lat1, lon1), (lat2, lon2)).meters / 1000
                if not (pd.isna(lat1) or pd.isna(lon1))
                else 0.0
                for lat1, lon1, lat2, lon2 in zip(
                    df["Latitud"].shift(), df["Longitud"].shift(),
                    df["Latitud"], df["Longitud"],
                )
            ]
            distancia_total = sum(distancias)
            # Ventana fija de 5 minutos (consistente con el entrenamiento)
            velocidad = distancia_total / (self._VENTANA_DISTANCIA_MIN / 60)
            return float(distancia_total), float(velocidad)
        except Exception as e:
            logger.warning(f"Error al calcular distancia/velocidad: {e}")
            return 0.0, 0.0
 
    def _predecir_time_to_park(
        self,
        area: float,
        var_vel: float,
        amplitud: float,
        distancia: float,
        vel_ventana: float,
    ) -> float:
        """
        Aplica el modelo lineal (con transformación logarítmica) para estimar
        el tiempo hasta aparcar en segundos.
        """
        log_pred = (
            self._COEF_INTERCEPT
            + self._COEF_AREA       * area
            + self._COEF_VAR_VEL   * var_vel
            + self._COEF_AMPLITUD  * amplitud
            + self._COEF_DISTANCIA * distancia
            + self._COEF_VEL_VENTANA * vel_ventana
        )
        return float(np.exp(log_pred))
 
    # ------------------------------------------------------------------
    # Zonas frecuentes
    # ------------------------------------------------------------------
 
    def _cargar_puntos_frecuentes(self, conductor_id: str) -> pd.DataFrame:
        """Carga los puntos frecuentes de un conductor con caché en memoria."""
        if conductor_id not in self._cache_pf:
            path = os.path.join(self.pf_folder, f"{conductor_id}_pf.csv")
            if not os.path.exists(path):
                logger.warning(
                    f"No se encontraron puntos frecuentes para {conductor_id}"
                )
                self._cache_pf[conductor_id] = pd.DataFrame(
                    columns=["Latitud", "Longitud"]
                )
            else:
                self._cache_pf[conductor_id] = pd.read_csv(path)
        return self._cache_pf[conductor_id]
 
    def _es_zona_frecuente(
        self, conductor_id: str, lat: float, lon: float
    ) -> bool:
        """
        Determina si las coordenadas actuales están dentro de una zona
        frecuente del conductor (radio de 250 m).
        """
        puntos = self._cargar_puntos_frecuentes(conductor_id)
        if puntos.empty:
            return False
 
        d = self._DISTANCIA_GRADOS
        mask = (
            (puntos["Latitud"] >= lat - d) & (puntos["Latitud"] <= lat + d) &
            (puntos["Longitud"] >= lon - d) & (puntos["Longitud"] <= lon + d)
        )
        return bool(mask.any())
 
    # ------------------------------------------------------------------
    # Evaluación de confianza
    # ------------------------------------------------------------------
 
    def _evaluar_confianza(
        self,
        vel_ventana: float,
        zona_frec: bool,
        area_cubierta: float,
        var_velocidad: float,
    ) -> str:
        """
        Evalúa la confianza de la predicción basándose en las condiciones
        del modelo. La confianza es mayor cuando el vehículo va lento,
        no está en zona frecuente y tiene un área cubierta pequeña
        (señal de que está buscando aparcamiento).
 
        Basado en el análisis del notebook e06.1: el MAE es significativamente
        menor cuando la velocidad ventana es menor de 40 km/h.
        """
        if zona_frec:
            return "baja"  # Zona frecuente → el modelo no es fiable aquí
        if vel_ventana > 75:
            return "baja"  # Velocidad alta → no está buscando aparcamiento
        if vel_ventana <= 40 and area_cubierta > 0 and var_velocidad < 30:
            return "alta"
        return "media"
 
 
# ---------------------------------------------------------------------------
# Módulo 3: Motor de integración — ETA Holístico
# ---------------------------------------------------------------------------
 
@dataclass
class OpcionParking:
    """Candidato a parking con su disponibilidad."""
    id: str
    nombre: str
    coords: tuple[float, float]   # (latitud, longitud)
    ocupacion: float              # 0.0 - 1.0
 
 
class ETAEngine:
    """
    Motor de integración que combina la predicción de tráfico con NEXTPARK
    para calcular un ETA holístico completo.
 
    Fórmula:
        ETA = T_conducción + T_maniobra + T_caminata
 
    Donde:
        - T_conducción: predicho por el modelo LSTM de tráfico
        - T_maniobra:   tiempo constante de rampa + maniobra de aparcamiento
        - T_caminata:   calculado a partir de la distancia parking → destino
    """
 
    # Parámetros operativos
    VELOCIDAD_CAMINATA_KMH  = 4.8    # Velocidad media a pie (km/h)
    TIEMPO_MANIOBRA_MIN     = 4.0    # Tiempo de rampa + maniobra de aparcamiento (min)
    UMBRAL_OCUPACION        = 0.95   # Rechazar parkings con ocupación >= 95%
    RADIO_BUSQUEDA_M        = 500.0  # Radio de búsqueda de parkings (m)
 
    def __init__(
        self,
        traffic_predictor: TrafficPredictor,
        nextpark_predictor: NextParkPredictor,
    ):
        self.traffic = traffic_predictor
        self.nextpark = nextpark_predictor
 
    def calcular_eta(
        self,
        conductor_id: str,
        node_id_destino: str,
        historial_viaje: pd.DataFrame,
        parkings_disponibles: list[OpcionParking],
        historial_nodo: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
    ) -> list[ETAHolistico]:
        """
        Calcula el ETA holístico para cada opción de parking disponible.
 
        Args:
            conductor_id: ID del conductor (para zonas frecuentes NEXTPARK).
            node_id_destino: ID del nodo de destino en la red de tráfico.
            historial_viaje: DataFrame con el historial del viaje actual
                             (columnas: Latitud, Longitud, FechaRTC).
            parkings_disponibles: Lista de parkings candidatos en el área de destino.
            historial_nodo: Historial del nodo de tráfico para el modelo LSTM.
                            Si es None, se usa modo simulación.
            timestamp: Momento del cálculo. Si es None, usa datetime.now().
 
        Returns:
            Lista de ETAHolistico ordenada por tiempo total ascendente,
            filtrando los parkings con ocupación crítica.
        """
        if timestamp is None:
            timestamp = datetime.now()
 
        logger.info(
            f"Calculando ETA holístico para conductor={conductor_id}, "
            f"destino={node_id_destino}, "
            f"parkings candidatos={len(parkings_disponibles)}"
        )
 
        # 1. Predicción de tráfico
        traffic_pred = self.traffic.predecir(
            node_id=node_id_destino,
            historial_nodo=historial_nodo,
            timestamp=timestamp,
        )
        logger.info(
            f"Tráfico: intensidad={traffic_pred.predicted_intensity:.0f} veh/h, "
            f"velocidad={traffic_pred.predicted_speed_kmh:.1f} km/h, "
            f"tiempo conducción={traffic_pred.travel_time_min:.1f} min"
        )
 
        # 2. Señal de aparcamiento NEXTPARK
        parking_signal = self.nextpark.generar_senal(
            historial_viaje=historial_viaje,
            conductor_id=conductor_id,
            timestamp=timestamp,
        )
        logger.info(
            f"NEXTPARK: aparca_en_10min={parking_signal.aparca_en_10min}, "
            f"time_to_park={parking_signal.time_to_park_seg:.0f}s, "
            f"confianza={parking_signal.confianza}"
        )
 
        # 3. Obtener posición actual del vehículo
        historial_viaje_sorted = historial_viaje.copy()
        historial_viaje_sorted["FechaRTC"] = pd.to_datetime(
            historial_viaje_sorted["FechaRTC"]
        )
        historial_viaje_sorted = historial_viaje_sorted.sort_values("FechaRTC")
        pos_actual = (
            float(historial_viaje_sorted.iloc[-1]["Latitud"]),
            float(historial_viaje_sorted.iloc[-1]["Longitud"]),
        )
 
        # 4. Calcular ETA para cada parking
        resultados = []
        for parking in parkings_disponibles:
            # Filtrar por ocupación crítica
            if parking.ocupacion >= self.UMBRAL_OCUPACION:
                logger.info(
                    f"Parking '{parking.nombre}' descartado: "
                    f"ocupación crítica ({parking.ocupacion * 100:.0f}%)"
                )
                continue
 
            # T_maniobra: constante
            t_maniobra = self.TIEMPO_MANIOBRA_MIN
 
            # T_caminata: distancia parking → destino a pie
            t_caminata, dist_m = self._calcular_caminata(
                parking.coords, pos_actual
            )
 
            # ETA total
            eta_total = (
                traffic_pred.travel_time_min
                + t_maniobra
                + t_caminata
            )
 
            resultado = ETAHolistico(
                parking_id=parking.id,
                parking_nombre=parking.nombre,
                parking_coords=parking.coords,
                parking_ocupacion=parking.ocupacion,
                t_conduccion_min=traffic_pred.travel_time_min,
                t_maniobra_min=t_maniobra,
                t_caminata_min=t_caminata,
                eta_total_min=eta_total,
                distancia_caminata_m=dist_m,
            )
            resultados.append(resultado)
            logger.info(resultado.resumen())
 
        if not resultados:
            logger.warning(
                "No se encontraron opciones de parking viables con los "
                "criterios actuales. Considera ampliar el radio de búsqueda "
                f"(actual: {self.RADIO_BUSQUEDA_M} m) o reducir el umbral "
                f"de ocupación (actual: {self.UMBRAL_OCUPACION * 100:.0f}%)."
            )
            return []
 
        # Ordenar por ETA total ascendente
        resultados.sort(key=lambda x: x.eta_total_min)
 
        mejor = resultados[0]
        logger.info(
            f"Mejor opción: {mejor.parking_nombre} — "
            f"ETA total: {mejor.eta_total_min:.1f} min"
        )
 
        return resultados
 
    def _calcular_caminata(
        self,
        coords_parking: tuple[float, float],
        coords_destino: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Calcula el tiempo de caminata y la distancia entre el parking y el destino.
 
        Args:
            coords_parking: (lat, lon) del parking.
            coords_destino: (lat, lon) del destino final (posición actual del vehículo
                            o destino explícito del usuario).
 
        Returns:
            (tiempo_caminata_min, distancia_m)
        """
        dist_m = geodesic(coords_parking, coords_destino).meters
        tiempo_min = (dist_m / 1000) / self.VELOCIDAD_CAMINATA_KMH * 60
        return float(tiempo_min), float(dist_m)
 
 
# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import json
 
    logger.info("=== Iniciando motor de integración PERTE VEC ===")
 
    # Inicializar predictores
    # En producción: TrafficPredictor(model_path="./modelos/lstm_cíclico.h5")
    traffic_predictor  = TrafficPredictor(model_path=None)   # Modo simulación
    nextpark_predictor = NextParkPredictor(pf_folder="./pf")
 
    engine = ETAEngine(
        traffic_predictor=traffic_predictor,
        nextpark_predictor=nextpark_predictor,
    )
 
    # ---- Datos de entrada ----
 
    # Historial del viaje actual (últimos 20 min del vehículo)
    ahora = datetime.now()
    historial_viaje = pd.DataFrame({
        "IdCliente":  ["P001"] * 6,
        "Latitud":    [40.418, 40.417, 40.416, 40.416, 40.415, 40.415],
        "Longitud":   [-3.703, -3.703, -3.703, -3.702, -3.702, -3.702],
        "FechaRTC":   pd.date_range(end=ahora, periods=6, freq="3min"),
        "TipoEvento": ["movement_tracking"] * 5 + ["movement_tracking"],
    })
 
    # Parkings disponibles en el área de destino
    parkings = [
        OpcionParking("PK-01", "Parking Centro",    (40.415, -3.702), ocupacion=0.75),
        OpcionParking("PK-02", "Parking Recoletos", (40.418, -3.705), ocupacion=0.98),
        OpcionParking("PK-03", "Parking Atocha",    (40.412, -3.698), ocupacion=0.50),
    ]
 
    # ---- Calcular ETA ----
    resultados = engine.calcular_eta(
        conductor_id="P001",
        node_id_destino="6798",
        historial_viaje=historial_viaje,
        parkings_disponibles=parkings,
        timestamp=ahora,
    )
 
    # ---- Mostrar resultados ----
    print("\n" + "=" * 55)
    print("  RESULTADO — ETA HOLÍSTICO PERTE VEC")
    print("=" * 55)
 
    if not resultados:
        print("No se encontraron opciones viables.")
    else:
        for i, eta in enumerate(resultados, 1):
            print(f"\n  Opción {i}: {eta.parking_nombre}")
            print(f"    Ocupación:          {eta.parking_ocupacion * 100:.0f}%")
            print(f"    T. conducción:      {eta.t_conduccion_min:.1f} min")
            print(f"    T. maniobra:        {eta.t_maniobra_min:.1f} min")
            print(f"    T. caminata:        {eta.t_caminata_min:.1f} min "
                  f"({eta.distancia_caminata_m:.0f} m)")
            print(f"    ─────────────────────────────────")
            print(f"    ETA TOTAL:          {eta.eta_total_min:.1f} min")
 
        mejor = resultados[0]
        print(f"\n  ✓ Recomendación: {mejor.parking_nombre} "
              f"({mejor.eta_total_min:.1f} min)")
 
    print("=" * 55 + "\n")