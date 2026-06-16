"""
Motor de integración: Predicción de tráfico + NEXTPARK
=======================================================
Integra el modelo LSTM de predicción de tráfico (Laura) con el algoritmo
NEXTPARK de predicción de aparcamiento para calcular un ETA holístico:
 
    ETA_total = T_conducción + T_maniobra_aparcamiento + T_caminata_a_pie
 
Autores: Next Mobility Solutions
Proyecto: ETA HOLISTICO — Expediente VE2-010000-2023-75
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
 
# Logging
# 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NextParkIntegration")
 
 
# Dataclasses de resultado
 
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
 
 
# Módulo 1: Predicción de tráfico (interfaz con el modelo LSTM de Laura)
 
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
 
 
# Módulo 2: Señal de aparcamiento NEXTPARK
 
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
 
    # API pública
 
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
 
    # Features del modelo
 
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
 
    # Zonas frecuentes
 
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
 
    # Evaluación de confianza
 
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
 
 
# Módulo 3: Motor de integración — ETA Holístico
 
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
 
 
# Ejemplo de uso
 
if __name__ == "__main__":
    import json
 
    logger.info("=== Iniciando motor de integración ETA HOLISTICO ===")
 
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
    print("  RESULTADO — ETA HOLÍSTICO ETA HOLISTICO")
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
 
 
# Módulo 4: Indicadores de experiencia de usuario (UX)
 
from enum import Enum
 
 
class SemaforoParkingColor(Enum):
    """
    Semáforo de dificultad de aparcamiento basado en el tiempo estimado
    hasta encontrar plaza (time_to_park_seg de NEXTPARK).
 
    Umbrales acordados con Next Mobility Solutions:
        VERDE    < 10 min  → Aparcamiento fácil
        AMARILLO 10-30 min → Aparcamiento medio
        ROJO     > 30 min  → Dificultad para aparcar
    """
    VERDE    = "🟢"
    AMARILLO = "🟡"
    ROJO     = "🔴"
 
 
class SemaforoTraficoConductor(Enum):
    """
    Semáforo de congestión de tráfico basado en la intensidad predicha
    por el modelo LSTM.
 
    Umbrales basados en el nivel de servicio del Ayuntamiento de Madrid:
        VERDE    < 800 veh/h   → Tráfico fluido
        AMARILLO 800-1500 veh/h → Tráfico moderado
        ROJO     > 1500 veh/h  → Congestión
    """
    VERDE    = "🟢"
    AMARILLO = "🟡"
    ROJO     = "🔴"
 
 
class NivelConfianzaColor(Enum):
    """
    Indicador visual del nivel de confianza de la predicción NEXTPARK.
    """
    ALTA   = "✅"
    MEDIA  = "⚠️"
    BAJA   = "❌"
 
 
@dataclass
class IndicadoresUX:
    """
    Conjunto completo de indicadores de experiencia de usuario.
    Diseñado para ser consumido directamente por la app móvil / CarPlay.
    """
    # --- Semáforos ---
    semaforo_parking: SemaforoParkingColor
    semaforo_trafico: SemaforoTraficoConductor
    confianza_prediccion: NivelConfianzaColor
 
    # --- Contadores ---
    tiempo_parking_min: float       # Contador inverso: tiempo hasta aparcar
    eta_total_min: float            # ETA holístico total
    distancia_caminata_m: float     # Metros a pie desde el parking al destino
 
    # --- Recomendación ---
    parking_recomendado: str        # Nombre del parking óptimo
    parking_ocupacion_pct: float    # % ocupación del parking recomendado
 
    # --- Alerta de zona frecuente ---
    zona_frecuente: bool            # True si el conductor está en una zona habitual
 
    def resumen_usuario(self) -> str:
        """
        Texto de resumen para mostrar al usuario en la app.
        Formato optimizado para pantalla de coche (CarPlay).
        """
        lineas = [
            f"╔══════════════════════════════════════╗",
            f"  NEXT MOBILITY — Asistente de Ruta",
            f"══════════════════════════════════════",
            f"",
            f"  TRÁFICO          {self.semaforo_trafico.value}  ETA conducción incluido",
            f"  APARCAMIENTO     {self.semaforo_parking.value}  {self.tiempo_parking_min:.0f} min estimados",
            f"  CONFIANZA        {self.confianza_prediccion.value}  Predicción {self._confianza_texto()}",
            f"",
            f"  ─────────────────────────────────────",
            f"  📍 Recomendado:  {self.parking_recomendado}",
            f"  🅿️  Ocupación:    {self.parking_ocupacion_pct:.0f}%",
            f"  🚶 A pie:        {self.distancia_caminata_m:.0f} m hasta tu destino",
            f"  ⏱️  ETA total:    {self.eta_total_min:.0f} min",
        ]
 
        if self.zona_frecuente:
            lineas.append(f"")
            lineas.append(f"  ℹ️  Zona habitual detectada")
 
        lineas.append(f"╚══════════════════════════════════════╝")
        return "\n".join(lineas)
 
    def _confianza_texto(self) -> str:
        mapping = {
            NivelConfianzaColor.ALTA:  "alta",
            NivelConfianzaColor.MEDIA: "media",
            NivelConfianzaColor.BAJA:  "baja",
        }
        return mapping[self.confianza_prediccion]
 
 
def calcular_semaforo_parking(time_to_park_seg: float) -> SemaforoParkingColor:
    """
    Calcula el color del semáforo de aparcamiento.
 
    Args:
        time_to_park_seg: Tiempo estimado hasta aparcar en segundos (NEXTPARK).
 
    Returns:
        SemaforoParkingColor según los umbrales definidos.
    """
    minutos = time_to_park_seg / 60
    if minutos < 10:
        return SemaforoParkingColor.VERDE
    elif minutos <= 30:
        return SemaforoParkingColor.AMARILLO
    else:
        return SemaforoParkingColor.ROJO
 
 
def calcular_semaforo_trafico(intensidad_veh_hora: float) -> SemaforoTraficoConductor:
    """
    Calcula el color del semáforo de tráfico.
 
    Args:
        intensidad_veh_hora: Intensidad predicha por el modelo LSTM (veh/hora).
 
    Returns:
        SemaforoTraficoConductor según el nivel de servicio.
    """
    if intensidad_veh_hora < 800:
        return SemaforoTraficoConductor.VERDE
    elif intensidad_veh_hora <= 1500:
        return SemaforoTraficoConductor.AMARILLO
    else:
        return SemaforoTraficoConductor.ROJO
 
 
def calcular_confianza_color(confianza: str) -> NivelConfianzaColor:
    """Convierte el texto de confianza de NEXTPARK en un indicador visual."""
    mapping = {
        "alta":  NivelConfianzaColor.ALTA,
        "media": NivelConfianzaColor.MEDIA,
        "baja":  NivelConfianzaColor.BAJA,
    }
    return mapping.get(confianza, NivelConfianzaColor.BAJA)
 
 
def generar_indicadores_ux(
    traffic_pred: TrafficPrediction,
    parking_signal: ParkingSignal,
    mejor_eta: ETAHolistico,
) -> IndicadoresUX:
    """
    Genera el conjunto completo de indicadores UX a partir de las predicciones.
 
    Args:
        traffic_pred:   Resultado del modelo LSTM de tráfico.
        parking_signal: Señal de aparcamiento de NEXTPARK.
        mejor_eta:      Mejor opción de ETA holístico calculada.
 
    Returns:
        IndicadoresUX listo para consumir por la app móvil / CarPlay.
    """
    return IndicadoresUX(
        semaforo_parking=calcular_semaforo_parking(parking_signal.time_to_park_seg),
        semaforo_trafico=calcular_semaforo_trafico(traffic_pred.predicted_intensity),
        confianza_prediccion=calcular_confianza_color(parking_signal.confianza),
        tiempo_parking_min=parking_signal.time_to_park_seg / 60,
        eta_total_min=mejor_eta.eta_total_min,
        distancia_caminata_m=mejor_eta.distancia_caminata_m,
        parking_recomendado=mejor_eta.parking_nombre,
        parking_ocupacion_pct=mejor_eta.parking_ocupacion * 100,
        zona_frecuente=parking_signal.zona_frecuente,
    )
 
 
# Demo completa con indicadores UX
 
def demo_completa():
    """
    Ejecuta una demo completa del motor con indicadores UX.
    Simula tres escenarios: hora valle, hora punta y aparcamiento difícil.
    """
    traffic_predictor  = TrafficPredictor(model_path=None)
    nextpark_predictor = NextParkPredictor(pf_folder="./pf")
    engine = ETAEngine(
        traffic_predictor=traffic_predictor,
        nextpark_predictor=nextpark_predictor,
    )
 
    escenarios = [
        {
            "nombre": "HORA VALLE (10:15h)",
            "hora": 10,
            "time_to_park_override": 412,   # ~7 min → VERDE
        },
        {
            "nombre": "HORA PUNTA (08:30h)",
            "hora": 8,
            "time_to_park_override": 1020,  # ~17 min → AMARILLO
        },
        {
            "nombre": "ZONA SATURADA (19:00h)",
            "hora": 19,
            "time_to_park_override": 2100,  # ~35 min → ROJO
        },
    ]
 
    ahora = datetime.now()
    historial_viaje = pd.DataFrame({
        "IdCliente":  ["P001"] * 6,
        "Latitud":    [40.418, 40.417, 40.416, 40.416, 40.415, 40.415],
        "Longitud":   [-3.703, -3.703, -3.703, -3.702, -3.702, -3.702],
        "FechaRTC":   pd.date_range(end=ahora, periods=6, freq="3min"),
        "TipoEvento": ["movement_tracking"] * 6,
    })
 
    parkings = [
        OpcionParking("PK-01", "Parking Centro",    (40.415, -3.702), ocupacion=0.75),
        OpcionParking("PK-02", "Parking Recoletos", (40.418, -3.705), ocupacion=0.98),
        OpcionParking("PK-03", "Parking Atocha",    (40.412, -3.698), ocupacion=0.50),
    ]
 
    for escenario in escenarios:
        print(f"\n{'='*55}")
        print(f"  ESCENARIO: {escenario['nombre']}")
        print(f"{'='*55}")
 
        # Predicción de tráfico
        ts = ahora.replace(hour=escenario["hora"], minute=0)
        traffic_pred = traffic_predictor.predecir("6798", timestamp=ts)
 
        # Señal NEXTPARK (simulamos el time_to_park del escenario)
        parking_signal = nextpark_predictor.generar_senal(
            historial_viaje=historial_viaje,
            conductor_id="P001",
            timestamp=ts,
        )
        # Override para la demo
        parking_signal.time_to_park_seg = escenario["time_to_park_override"]
 
        # ETA holístico
        resultados = engine.calcular_eta(
            conductor_id="P001",
            node_id_destino="6798",
            historial_viaje=historial_viaje,
            parkings_disponibles=parkings,
            timestamp=ts,
        )
 
        if not resultados:
            print("  Sin opciones viables.")
            continue
 
        mejor = resultados[0]
 
        # Indicadores UX
        ux = generar_indicadores_ux(traffic_pred, parking_signal, mejor)
        print(ux.resumen_usuario())
 
 
if __name__ == "__main__":
    demo_completa()
 
 
# Módulo 5: Indicadores de experiencia de usuario avanzados
 
@dataclass
class AlertaSalidaOptima:
    """
    Alerta de salida óptima: informa al usuario si saliendo en X minutos
    puede evitar tráfico y ahorrar tiempo de conducción.
 
    Basada en la predicción LSTM para distintas franjas horarias.
    """
    salir_ahora_min: float          # ETA si sale ahora mismo
    salir_en_x_min: float           # Minutos de espera recomendados
    eta_si_espera_min: float        # ETA si espera X minutos
    ahorro_min: float               # Minutos ahorrados esperando
    merece_esperar: bool            # True si el ahorro > tiempo de espera
 
    def mensaje(self) -> str:
        if not self.merece_esperar:
            return "🚗 Sal ahora, el tráfico no mejorará significativamente."
        return (
            f"⏰ Si esperas {self.salir_en_x_min:.0f} min, "
            f"ahorras {self.ahorro_min:.0f} min de tráfico."
        )
 
 
@dataclass
class IndicadorAhorroTiempo:
    """
    Compara el ETA del parking recomendado frente a las demás opciones
    y frente a aparcar en calle (estimación).
    """
    parking_recomendado: str
    eta_recomendado_min: float
    eta_segunda_opcion_min: float
    ahorro_vs_segunda_min: float
    ahorro_vs_calle_min: float      # Estimación vs aparcar en calle (~+15 min)
 
    def mensaje(self) -> str:
        lineas = [f"💡 {self.parking_recomendado} es tu mejor opción."]
        if self.ahorro_vs_segunda_min > 1:
            lineas.append(
                f"   Ahorras {self.ahorro_vs_segunda_min:.0f} min "
                f"frente al siguiente parking."
            )
        if self.ahorro_vs_calle_min > 0:
            lineas.append(
                f"   Ahorras ~{self.ahorro_vs_calle_min:.0f} min "
                f"frente a buscar aparcamiento en calle."
            )
        return "\n".join(lineas)
 
 
@dataclass
class SemaforoZonaDestino:
    """
    Semáforo de velocidad de la zona de destino basado en el mapa de
    velocidades máximas inferido sobre la red viaria de la Comunidad de Madrid
    (proyecto de inferencia de velocidades ETA HOLISTICO).
 
    Permite al conductor anticipar la velocidad máxima en la zona de maniobra.
    """
    velocidad_max_kmh: float        # Velocidad máxima de la zona (30/50/70/90...)
    es_zona_30: bool                # True si es zona de velocidad reducida
    semaforo: SemaforoParkingColor  # Reutilizamos el semáforo de colores
 
    def mensaje(self) -> str:
        if self.es_zona_30:
            return f"🐢 Zona 30 — Reduce la velocidad al llegar."
        elif self.velocidad_max_kmh <= 50:
            return f"🚗 Zona urbana — Velocidad máxima {self.velocidad_max_kmh:.0f} km/h."
        else:
            return f"🛣️ Vía rápida — Velocidad máxima {self.velocidad_max_kmh:.0f} km/h."
 
 
@dataclass
class AlertaBateria:
    """
    Alerta de batería basada en los datos OBD del vehículo.
    Estima si la batería actual es suficiente para llegar al destino
    más el parking más cercano disponible.
 
    Requiere acceso a los datos OBD del vehículo (nivel de batería en %).
    """
    bateria_actual_pct: float       # % de batería actual (fuente: OBD)
    autonomia_estimada_km: float    # Autonomía restante estimada (km)
    distancia_destino_km: float     # Distancia al destino (km)
    distancia_parking_km: float     # Distancia adicional al parking (km)
    distancia_total_km: float       # Distancia total necesaria (km)
    bateria_suficiente: bool        # True si la batería alcanza
    punto_recarga_cercano: bool     # True si hay recarga cerca del parking
    margen_bateria_pct: float       # % de batería sobrante tras el trayecto
 
    def semaforo(self) -> SemaforoParkingColor:
        if self.bateria_suficiente and self.margen_bateria_pct >= 20:
            return SemaforoParkingColor.VERDE
        elif self.bateria_suficiente and self.margen_bateria_pct < 20:
            return SemaforoParkingColor.AMARILLO
        else:
            return SemaforoParkingColor.ROJO
 
    def mensaje(self) -> str:
        if not self.bateria_suficiente:
            return (
                f"🔋 ATENCIÓN: Batería insuficiente ({self.bateria_actual_pct:.0f}%). "
                f"Necesitas recargar antes de llegar."
            )
        elif self.margen_bateria_pct < 20:
            return (
                f"🔋 Batería justa ({self.bateria_actual_pct:.0f}%). "
                f"Te quedará un {self.margen_bateria_pct:.0f}% al llegar."
            )
        else:
            return (
                f"🔋 Batería suficiente ({self.bateria_actual_pct:.0f}%). "
                f"Margen: {self.margen_bateria_pct:.0f}%."
            )
 
 
@dataclass
class IndicadorRecargaCercana:
    """
    Indica si hay un punto de recarga eléctrica cerca del parking recomendado
    y estima el tiempo de espera basándose en la ocupación del cargador.
    """
    hay_recarga_cercana: bool
    nombre_punto_recarga: Optional[str] = None
    distancia_recarga_m: Optional[float] = None
    tiempo_espera_min: Optional[float] = None       # Estimado por ocupación
    potencia_kw: Optional[float] = None             # Potencia del cargador (kW)
    tiempo_carga_30min_min: Optional[float] = None  # Tiempo para cargar 30% (min)
 
    def mensaje(self) -> str:
        if not self.hay_recarga_cercana:
            return "⚡ No hay puntos de recarga cercanos al parking recomendado."
        espera = f", espera estimada: {self.tiempo_espera_min:.0f} min" \
                 if self.tiempo_espera_min is not None else ""
        recarga = f", {self.tiempo_carga_30min_min:.0f} min para +30%" \
                  if self.tiempo_carga_30min_min is not None else ""
        return (
            f"⚡ {self.nombre_punto_recarga} a {self.distancia_recarga_m:.0f} m "
            f"({self.potencia_kw:.0f} kW{espera}{recarga})."
        )
 
 
@dataclass
class IndicadoresUXAvanzados:
    """
    Conjunto completo de indicadores UX avanzados.
    Extiende IndicadoresUX con las funcionalidades adicionales.
    """
    # Indicadores base
    base: IndicadoresUX
 
    # Indicadores avanzados
    alerta_salida: AlertaSalidaOptima
    ahorro_tiempo: IndicadorAhorroTiempo
    zona_destino: SemaforoZonaDestino
    bateria: AlertaBateria
    recarga: IndicadorRecargaCercana
 
    def resumen_usuario_completo(self) -> str:
        """
        Texto de resumen completo para la app móvil / CarPlay.
        """
        lineas = [
            f"╔══════════════════════════════════════╗",
            f"  NEXT MOBILITY — Asistente de Ruta",
            f"══════════════════════════════════════",
            f"",
            f"  TRÁFICO          {self.base.semaforo_trafico.value}  ETA conducción incluido",
            f"  APARCAMIENTO     {self.base.semaforo_parking.value}  {self.base.tiempo_parking_min:.0f} min estimados",
            f"  BATERÍA          {self.bateria.semaforo().value}  {self.bateria.mensaje()}",
            f"  ZONA DESTINO     {self.zona_destino.semaforo.value}  {self.zona_destino.mensaje()}",
            f"  CONFIANZA        {self.base.confianza_prediccion.value}  Predicción {self.base._confianza_texto()}",
            f"",
            f"  ─────────────────────────────────────",
            f"  📍 Recomendado:  {self.base.parking_recomendado}",
            f"  🅿️  Ocupación:    {self.base.parking_ocupacion_pct:.0f}%",
            f"  🚶 A pie:        {self.base.distancia_caminata_m:.0f} m hasta tu destino",
            f"  ⏱️  ETA total:    {self.base.eta_total_min:.0f} min",
            f"",
            f"  ─────────────────────────────────────",
            f"  {self.alerta_salida.mensaje()}",
            f"  {self.ahorro_tiempo.mensaje()}",
            f"  {self.recarga.mensaje()}",
        ]
 
        if self.base.zona_frecuente:
            lineas.append(f"  ℹ️  Zona habitual detectada")
 
        lineas.append(f"╚══════════════════════════════════════╝")
        return "\n".join(lineas)
 
 
# Funciones de cálculo de indicadores avanzados
 
def calcular_alerta_salida_optima(
    traffic_predictor: TrafficPredictor,
    node_id: str,
    timestamp: datetime,
    intervalos_min: list[int] = [15, 30, 45],
) -> AlertaSalidaOptima:
    """
    Calcula la alerta de salida óptima comparando el ETA actual con el ETA
    si el usuario espera 15, 30 o 45 minutos.
 
    Args:
        traffic_predictor: Instancia del predictor de tráfico.
        node_id: Nodo de destino.
        timestamp: Momento actual.
        intervalos_min: Intervalos de espera a evaluar (minutos).
 
    Returns:
        AlertaSalidaOptima con el intervalo óptimo de espera.
    """
    pred_ahora = traffic_predictor.predecir(node_id, timestamp=timestamp)
    eta_ahora = pred_ahora.travel_time_min
 
    mejor_ahorro = 0.0
    mejor_espera = 0
    mejor_eta = eta_ahora
 
    for espera in intervalos_min:
        ts_futuro = timestamp.replace(
            hour=(timestamp.hour + (timestamp.minute + espera) // 60) % 24,
            minute=(timestamp.minute + espera) % 60,
        )
        pred_futuro = traffic_predictor.predecir(node_id, timestamp=ts_futuro)
        ahorro = eta_ahora - pred_futuro.travel_time_min
 
        # Solo merece esperar si el ahorro es mayor que el tiempo de espera
        if ahorro > espera and ahorro > mejor_ahorro:
            mejor_ahorro = ahorro
            mejor_espera = espera
            mejor_eta = pred_futuro.travel_time_min
 
    merece_esperar = mejor_espera > 0 and mejor_ahorro > mejor_espera
 
    return AlertaSalidaOptima(
        salir_ahora_min=eta_ahora,
        salir_en_x_min=float(mejor_espera),
        eta_si_espera_min=mejor_eta,
        ahorro_min=mejor_ahorro,
        merece_esperar=merece_esperar,
    )
 
 
def calcular_ahorro_tiempo(
    resultados: list[ETAHolistico],
    tiempo_calle_extra_min: float = 15.0,
) -> IndicadorAhorroTiempo:
    """
    Calcula el ahorro de tiempo del parking recomendado frente a otras opciones
    y frente a aparcar en calle.
 
    Args:
        resultados: Lista de ETAHolistico ordenada por tiempo ascendente.
        tiempo_calle_extra_min: Tiempo extra estimado para aparcar en calle (min).
 
    Returns:
        IndicadorAhorroTiempo con el ahorro calculado.
    """
    if not resultados:
        return IndicadorAhorroTiempo(
            parking_recomendado="N/A",
            eta_recomendado_min=0.0,
            eta_segunda_opcion_min=0.0,
            ahorro_vs_segunda_min=0.0,
            ahorro_vs_calle_min=0.0,
        )
 
    mejor = resultados[0]
    segunda = resultados[1] if len(resultados) > 1 else None
    eta_segunda = segunda.eta_total_min if segunda else mejor.eta_total_min
    eta_calle = mejor.t_conduccion_min + tiempo_calle_extra_min
 
    return IndicadorAhorroTiempo(
        parking_recomendado=mejor.parking_nombre,
        eta_recomendado_min=mejor.eta_total_min,
        eta_segunda_opcion_min=eta_segunda,
        ahorro_vs_segunda_min=max(0.0, eta_segunda - mejor.eta_total_min),
        ahorro_vs_calle_min=max(0.0, eta_calle - mejor.eta_total_min),
    )
 
 
def calcular_semaforo_zona_destino(
    velocidad_max_kmh: float,
) -> SemaforoZonaDestino:
    """
    Calcula el semáforo de zona de destino a partir de la velocidad máxima
    inferida por el mapa de velocidades del proyecto ETA HOLISTICO.
 
    Args:
        velocidad_max_kmh: Velocidad máxima del tramo de destino (km/h).
                           Obtenida del dataset GeoJSON de velocidades.
 
    Returns:
        SemaforoZonaDestino con el color y mensaje correspondientes.
    """
    es_zona_30 = velocidad_max_kmh <= 30
 
    if velocidad_max_kmh <= 30:
        color = SemaforoParkingColor.ROJO
    elif velocidad_max_kmh <= 50:
        color = SemaforoParkingColor.AMARILLO
    else:
        color = SemaforoParkingColor.VERDE
 
    return SemaforoZonaDestino(
        velocidad_max_kmh=velocidad_max_kmh,
        es_zona_30=es_zona_30,
        semaforo=color,
    )
 
 
def calcular_alerta_bateria(
    bateria_actual_pct: float,
    distancia_destino_km: float,
    distancia_parking_km: float,
    autonomia_total_km: float = 400.0,
    hay_recarga_cercana: bool = False,
) -> AlertaBateria:
    """
    Calcula la alerta de batería estimando si la autonomía restante es
    suficiente para llegar al destino más el parking.
 
    Args:
        bateria_actual_pct: % de batería actual (fuente: OBD).
        distancia_destino_km: Distancia al destino en km.
        distancia_parking_km: Distancia adicional al parking en km.
        autonomia_total_km: Autonomía total del vehículo al 100% (km).
        hay_recarga_cercana: True si hay un cargador cerca del parking.
 
    Returns:
        AlertaBateria con el estado de la batería y el semáforo.
    """
    autonomia_restante = autonomia_total_km * (bateria_actual_pct / 100)
    distancia_total = distancia_destino_km + distancia_parking_km
    bateria_suficiente = autonomia_restante >= distancia_total
 
    # Batería residual tras el trayecto
    bateria_residual_km = autonomia_restante - distancia_total
    margen_pct = max(0.0, (bateria_residual_km / autonomia_total_km) * 100)
 
    return AlertaBateria(
        bateria_actual_pct=bateria_actual_pct,
        autonomia_estimada_km=autonomia_restante,
        distancia_destino_km=distancia_destino_km,
        distancia_parking_km=distancia_parking_km,
        distancia_total_km=distancia_total,
        bateria_suficiente=bateria_suficiente,
        punto_recarga_cercano=hay_recarga_cercana,
        margen_bateria_pct=margen_pct,
    )
 
 
def calcular_indicador_recarga(
    parking_coords: tuple[float, float],
    puntos_recarga: list[dict],
    radio_busqueda_m: float = 300.0,
) -> IndicadorRecargaCercana:
    """
    Busca el punto de recarga más cercano al parking recomendado.
 
    Args:
        parking_coords: (lat, lon) del parking recomendado.
        puntos_recarga: Lista de puntos de recarga con campos:
                        nombre, coords (lat, lon), potencia_kw, ocupacion (0-1).
        radio_busqueda_m: Radio de búsqueda en metros.
 
    Returns:
        IndicadorRecargaCercana con el punto más cercano (si existe).
    """
    candidatos = []
    for punto in puntos_recarga:
        dist_m = geodesic(parking_coords, punto["coords"]).meters
        if dist_m <= radio_busqueda_m:
            candidatos.append({**punto, "distancia_m": dist_m})
 
    if not candidatos:
        return IndicadorRecargaCercana(hay_recarga_cercana=False)
 
    # Seleccionar el más cercano con menor ocupación
    candidatos.sort(key=lambda x: (x["distancia_m"], x.get("ocupacion", 0)))
    mejor = candidatos[0]
 
    # Estimar tiempo de espera basado en ocupación
    ocupacion = mejor.get("ocupacion", 0.0)
    tiempo_espera = 0.0 if ocupacion < 0.5 else (10.0 if ocupacion < 0.8 else 20.0)
 
    # Estimar tiempo para cargar 30% a la potencia del cargador
    # Batería media VEC: 75 kWh → 30% = 22.5 kWh
    potencia = mejor.get("potencia_kw", 22.0)
    tiempo_carga_30 = (22.5 / potencia) * 60  # minutos
 
    return IndicadorRecargaCercana(
        hay_recarga_cercana=True,
        nombre_punto_recarga=mejor["nombre"],
        distancia_recarga_m=mejor["distancia_m"],
        tiempo_espera_min=tiempo_espera,
        potencia_kw=potencia,
        tiempo_carga_30min_min=tiempo_carga_30,
    )
 
 
def generar_indicadores_ux_avanzados(
    traffic_predictor: TrafficPredictor,
    traffic_pred: TrafficPrediction,
    parking_signal: ParkingSignal,
    resultados: list[ETAHolistico],
    velocidad_max_zona_kmh: float,
    bateria_actual_pct: float,
    distancia_destino_km: float,
    puntos_recarga: list[dict],
    timestamp: Optional[datetime] = None,
) -> IndicadoresUXAvanzados:
    """
    Genera el conjunto completo de indicadores UX avanzados.
 
    Args:
        traffic_predictor: Instancia del predictor de tráfico.
        traffic_pred: Predicción de tráfico ya calculada.
        parking_signal: Señal de aparcamiento NEXTPARK.
        resultados: Lista de ETAHolistico ordenada por tiempo.
        velocidad_max_zona_kmh: Velocidad máxima de la zona de destino.
        bateria_actual_pct: % de batería actual (OBD).
        distancia_destino_km: Distancia al destino en km.
        puntos_recarga: Lista de puntos de recarga cercanos.
        timestamp: Momento del cálculo.
 
    Returns:
        IndicadoresUXAvanzados con todos los indicadores calculados.
    """
    if timestamp is None:
        timestamp = datetime.now()
 
    mejor = resultados[0]
 
    # Indicadores base
    base = generar_indicadores_ux(traffic_pred, parking_signal, mejor)
 
    # Alerta de salida óptima
    alerta_salida = calcular_alerta_salida_optima(
        traffic_predictor, traffic_pred.node_id, timestamp
    )
 
    # Ahorro de tiempo
    ahorro = calcular_ahorro_tiempo(resultados)
 
    # Semáforo zona destino
    zona = calcular_semaforo_zona_destino(velocidad_max_zona_kmh)
 
    # Alerta batería
    dist_parking_km = mejor.distancia_caminata_m / 1000
    bateria = calcular_alerta_bateria(
        bateria_actual_pct=bateria_actual_pct,
        distancia_destino_km=distancia_destino_km,
        distancia_parking_km=dist_parking_km,
        hay_recarga_cercana=len(puntos_recarga) > 0,
    )
 
    # Indicador recarga
    recarga = calcular_indicador_recarga(mejor.parking_coords, puntos_recarga)
 
    return IndicadoresUXAvanzados(
        base=base,
        alerta_salida=alerta_salida,
        ahorro_tiempo=ahorro,
        zona_destino=zona,
        bateria=bateria,
        recarga=recarga,
    )
 
 
# Demo completa con indicadores UX avanzados
 
def demo_avanzada():
    """
    Demo completa con todos los indicadores UX avanzados.
    Simula tres escenarios: hora valle, hora punta y zona saturada.
    """
    traffic_predictor  = TrafficPredictor(model_path=None)
    nextpark_predictor = NextParkPredictor(pf_folder="./pf")
    engine = ETAEngine(
        traffic_predictor=traffic_predictor,
        nextpark_predictor=nextpark_predictor,
    )
 
    # Puntos de recarga cercanos al área de destino
    puntos_recarga = [
        {
            "nombre": "Recarga Centro Comercial",
            "coords": (40.414, -3.701),
            "potencia_kw": 22.0,
            "ocupacion": 0.3,
        },
        {
            "nombre": "Recarga Parking Atocha",
            "coords": (40.411, -3.697),
            "potencia_kw": 50.0,
            "ocupacion": 0.7,
        },
    ]
 
    escenarios = [
        {
            "nombre": "HORA VALLE (10:15h)",
            "hora": 10,
            "time_to_park_override": 412,    # ~7 min → VERDE
            "bateria_pct": 75.0,
            "velocidad_zona": 30.0,
        },
        {
            "nombre": "HORA PUNTA (08:30h)",
            "hora": 8,
            "time_to_park_override": 1020,   # ~17 min → AMARILLO
            "bateria_pct": 25.0,
            "velocidad_zona": 50.0,
        },
        {
            "nombre": "ZONA SATURADA (19:00h)",
            "hora": 19,
            "time_to_park_override": 2100,   # ~35 min → ROJO
            "bateria_pct": 60.0,
            "velocidad_zona": 70.0,
        },
    ]
 
    ahora = datetime.now()
    historial_viaje = pd.DataFrame({
        "IdCliente":  ["P001"] * 6,
        "Latitud":    [40.418, 40.417, 40.416, 40.416, 40.415, 40.415],
        "Longitud":   [-3.703, -3.703, -3.703, -3.702, -3.702, -3.702],
        "FechaRTC":   pd.date_range(end=ahora, periods=6, freq="3min"),
        "TipoEvento": ["movement_tracking"] * 6,
    })
 
    parkings = [
        OpcionParking("PK-01", "Parking Centro",    (40.415, -3.702), ocupacion=0.75),
        OpcionParking("PK-02", "Parking Recoletos", (40.418, -3.705), ocupacion=0.98),
        OpcionParking("PK-03", "Parking Atocha",    (40.412, -3.698), ocupacion=0.50),
    ]
 
    for escenario in escenarios:
        print(f"\n{'='*55}")
        print(f"  ESCENARIO: {escenario['nombre']}")
        print(f"{'='*55}")
 
        ts = ahora.replace(hour=escenario["hora"], minute=0)
 
        # Predicción de tráfico
        traffic_pred = traffic_predictor.predecir("6798", timestamp=ts)
 
        # Señal NEXTPARK
        parking_signal = nextpark_predictor.generar_senal(
            historial_viaje=historial_viaje,
            conductor_id="P001",
            timestamp=ts,
        )
        parking_signal.time_to_park_seg = escenario["time_to_park_override"]
 
        # ETA holístico
        resultados = engine.calcular_eta(
            conductor_id="P001",
            node_id_destino="6798",
            historial_viaje=historial_viaje,
            parkings_disponibles=parkings,
            timestamp=ts,
        )
 
        if not resultados:
            print("  Sin opciones viables.")
            continue
 
        # Indicadores UX avanzados
        ux = generar_indicadores_ux_avanzados(
            traffic_predictor=traffic_predictor,
            traffic_pred=traffic_pred,
            parking_signal=parking_signal,
            resultados=resultados,
            velocidad_max_zona_kmh=escenario["velocidad_zona"],
            bateria_actual_pct=escenario["bateria_pct"],
            distancia_destino_km=5.0,
            puntos_recarga=puntos_recarga,
            timestamp=ts,
        )
 
        print(ux.resumen_usuario_completo())
 
 
if __name__ == "__main__":
    demo_avanzada()