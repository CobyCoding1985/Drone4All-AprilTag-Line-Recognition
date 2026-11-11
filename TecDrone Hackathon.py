import time
import math
import logging
import threading
import cv2
import numpy as np
import pupil_apriltags as apriltag
from djitellopy import Tello

# CONFIGURACIÓN Y PARÁMETROS
INVERTIR_ESPEJO = True        
ROTAR_180 = False             

# --- Control de Vuelo y Línea ---
ALTURA_OBJETIVO = 50          # cm
VEL_AVANCE = 15               
KP_GIRO = 0.8                 
MAX_GIRO = 35                 
KP_LADO = 0.12                
MAX_LADO = 20                 

# --- Detección de la línea negra ---
UMBRAL_NEGRO = 75             
MIN_AREA_LINEA = 800          
MARGEN_INFERIOR = 120         # Ignora la parte baja (espejo/dron)
MARGEN_SUPERIOR = 0           

# --- AprilTags ---
FAMILIA = "tag36h11"          
MIN_MARGEN = 20               
ZONA = (40, 20, 600, 460)     # Zona de aceptación en resolución 640x480
Y_DISPARO_TAG = 50            
TIEMPO_COOLDOWN_POST = 12.0    
MIN_AREA_TAG = 3000           

detector = apriltag.Detector(families=FAMILIA, nthreads=2)

def preparar_imagen(frame):
    frame = cv2.resize(frame, (640, 480))
    if INVERTIR_ESPEJO:
        frame = cv2.flip(frame, 0)
    if ROTAR_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def detectar_tags(gray):
    crudos = detector.detect(gray)
    x1, y1, x2, y2 = ZONA
    validos = []
    for t in crudos:
        cx, cy = t.center
        if (t.tag_id in [1, 2, 3, 4] and t.hamming == 0 and t.decision_margin >= MIN_MARGEN
                and x1 <= cx <= x2 and y1 <= cy <= y2):
            validos.append(t)
    return crudos, validos

def area_tag(t):
    return cv2.contourArea(t.corners.astype(np.float32))

def mascara_linea(gray, tags_crudos):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binaria = cv2.threshold(blur, UMBRAL_NEGRO, 255, cv2.THRESH_BINARY_INV)

    k = np.ones((5, 5), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, k)
    
    h, w = binaria.shape
    if MARGEN_INFERIOR > 0:
        binaria[h - MARGEN_INFERIOR:, :] = 0
    if MARGEN_SUPERIOR > 0:
        binaria[:MARGEN_SUPERIOR, :] = 0
        
    for t in tags_crudos:  
        c = t.center
        pts = (c + (t.corners - c) * 1.4).astype(np.int32)
        cv2.fillConvexPoly(binaria, pts, 0)
        
    return binaria

def elegir_contorno(binaria, punto_ref):
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidatos = [c for c in contornos if cv2.contourArea(c) >= MIN_AREA_LINEA]
    if not candidatos:
        return None
    px, py = float(punto_ref[0]), float(punto_ref[1])

    def costo(c):
        d = cv2.pointPolygonTest(c, (px, py), True)
        return 0.0 if d >= 0 else -d

    return min(candidatos, key=costo)

def centro_franja(mask, y0, y1):
    franja = mask[y0:y1, :]
    m = cv2.moments(franja, binaryImage=True)
    if m["m00"] < 150:
        return None
    return (m["m10"] / m["m00"], y0 + m["m01"] / m["m00"])

def analizar_linea(binaria, punto_ref):
    h, w = binaria.shape
    c = elegir_contorno(binaria, punto_ref)
    if c is None:
        return None
    solo = np.zeros_like(binaria)
    cv2.drawContours(solo, [c], -1, 255, cv2.FILLED)

    top = MARGEN_SUPERIOR
    alto = (h - MARGEN_INFERIOR) - top
    lejos = centro_franja(solo, top + int(0.10 * alto), top + int(0.40 * alto))
    cerca = centro_franja(solo, top + int(0.60 * alto), top + int(0.95 * alto))

    if cerca is not None and lejos is not None:
        angulo = math.degrees(math.atan2(lejos[0] - cerca[0], cerca[1] - lejos[1]))
    else:
        vx, vy, _, _ = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if vy > 0:  
            vx, vy = -vx, -vy
        angulo = math.degrees(math.atan2(vx, -vy))

    m = cv2.moments(c)
    centro = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (w / 2, h / 2)
    base = cerca if cerca is not None else centro
    return {
        "contorno": c,
        "angulo": angulo,
        "desplazamiento": base[0] - w / 2,
        "cerca": cerca,
        "lejos": lejos,
        "ref": base,
    }

def limitar(v, m):
    return int(max(-m, min(m, v)))

# PROGRAMA PRINCIPAL
def main():
    Tello.LOGGER.setLevel(logging.WARNING)
    tello = Tello()
    print("Conectando con el Tello...")
    tello.connect()
    print(f"Batería actual: {tello.get_battery()}%")

    try:
        tello.streamon()
        time.sleep(2.0)  
        frame_read = tello.get_frame_read()
    except Exception as e:
        print(f"Error al iniciar video: {e}")
        return

    is_flying = False
    status_msg = "LISTO - Presiona ENTER para despegar"
    ref_linea = None
    tiempo_fin_bloqueo = 0.0

    trabajo = {"hilo": None, "desc": ""}
    def ocupado():
        return trabajo["hilo"] is not None and trabajo["hilo"].is_alive()

    def en_fondo(desc, funcion):
        def correr():
            nonlocal tiempo_fin_bloqueo
            try:
                funcion()
            except Exception as e:
                print(f"Aviso en '{desc}': {e}")
            tiempo_fin_bloqueo = time.time() + TIEMPO_COOLDOWN_POST
        
        trabajo["desc"] = desc
        trabajo["hilo"] = threading.Thread(target=correr, daemon=True)
        trabajo["hilo"].start()

    def rc(lr, fb, ud, yaw):
        if is_flying and not ocupado():
            tello.send_rc_control(int(lr), int(fb), int(ud), int(yaw))

    accion_pendiente = None
    etapa = "PREVUELO"
    
    def cambiar(nueva, motivo=""):
        nonlocal etapa
        etapa = nueva
        print(f"ETAPA -> {nueva}" + (f" ({motivo})" if motivo else ""))

    prev_time = time.time()
    
    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                continue
            
            img = preparar_imagen(frame)
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            crudos, tags = detectar_tags(gray)
            binaria = mascara_linea(gray, crudos)
            
            if ref_linea is None:
                ref_linea = (w / 2, h * 0.75)
            
            linea = analizar_linea(binaria, ref_linea)
            vista = img.copy()

            ahora = time.time()
            fps = 1.0 / max(ahora - prev_time, 1e-3)
            prev_time = ahora

            if ocupado():
                status_msg = f"Ejecutando: {trabajo['desc']}..."

            elif etapa == "PREVUELO":
                status_msg = "Presiona ENTER para despegar | Q para salir"

            elif etapa == "SEGUIR LINEA":
                if linea is not None:
                    ref_linea = linea["ref"]

                en_tiempo_ciego = (time.time() < tiempo_fin_bloqueo)

                td = None
                
                # 1. Buscamos primero de inmediato el ID 3 sin importar el tiempo ciego
                for t in tags:
                    if t.tag_id == 3 and area_tag(t) >= 1000:
                        td = t
                        break

                # 2. Si no está el ID 3, evaluamos los demás tags respetando el tiempo ciego y filtros
                if td is None and not en_tiempo_ciego:
                    for t in sorted(tags, key=area_tag, reverse=True):
                        tid = t.tag_id
                        if tid in (1, 2, 4) and area_tag(t) >= MIN_AREA_TAG:
                            td = t
                            break

                if td is not None and (td.tag_id == 3 or td.center[1] >= Y_DISPARO_TAG or linea is None):
                    rc(0, 0, 0, 0)
                    accion_pendiente = td.tag_id
                    
                    if td.tag_id == 3:
                        cambiar("ALINEAR_ATERRIZAJE", "¡ID 3 META detectado! Iniciando alineación para aterrizar")
                    else:
                        cambiar("ACCION", f"Tag ID{td.tag_id} detectado")

                elif linea is not None:
                    a = linea["angulo"]
                    off = linea["desplazamiento"]
                    
                    yaw = limitar(KP_GIRO * a, MAX_GIRO)
                    lr = limitar(KP_LADO * off, MAX_LADO)
                    
                    factor_velocidad = max(0.2, 1.0 - (abs(a) / 45.0) - (abs(off) / (w / 2)))
                    fb = VEL_AVANCE * factor_velocidad

                    rc(lr, fb, 0, yaw)

                    if en_tiempo_ciego:
                        status_msg = f"Alejándose del Tag (Tiempo ciego)..."
                    else:
                        status_msg = f"Siguiendo Línea | Ang:{a:.0f} Off:{off:.0f}"
                    
                    cv2.drawContours(vista, [linea["contorno"]], -1, (0, 255, 0), 2)
                else:
                    rc(0, VEL_AVANCE, 0, 0)
                    status_msg = "Buscando nueva pista..."

            elif etapa == "ALINEAR_ATERRIZAJE":
                tag_3 = next((t for t in tags if t.tag_id == 3), None)
                if tag_3 is not None:
                    cx, cy = tag_3.center
                    error_x = cx - (w / 2)
                    error_y = cy - (h / 2)
                    
                    if abs(error_x) < 25 and abs(error_y) < 25:
                        rc(0, 0, 0, 0)
                        status_msg = "Alineado con ID 3 - Aterrizando..."
                        print("¡Alineación con ID 3 completada! Aterrizando...")
                        try:
                            tello.send_rc_control(0, 0, 0, 0)
                            tello.land()
                        except Exception:
                            pass
                        is_flying = False
                        break
                    else:
                        lr_cmd = limitar(error_x * 0.1, 12)
                        fb_cmd = limitar(-error_y * 0.1, 12)
                        rc(lr_cmd, fb_cmd, 0, 0)
                        status_msg = f"Alineando ID3... ErrX:{error_x:.0f} ErrY:{error_y:.0f}"
                else:
                    rc(0, 0, 0, 0)
                    status_msg = "Buscando ID 3 para alinear (Frenado)..."

            elif etapa == "ACCION":
                rc(0, 0, 0, 0)
                tid = accion_pendiente
                status_msg = f"Ejecutando Accion ID {tid}..."
                
                if trabajo["hilo"] is None:
                    if tid == 1:
                        en_fondo("Avanzar ID1", lambda: (tello.move_forward(50), time.sleep(0.5)))
                    elif tid == 2:
                        en_fondo("Giro Izq ID2", lambda: (tello.rotate_counter_clockwise(90), tello.move_forward(50), time.sleep(0.5)))
                    elif tid == 4:
                        en_fondo("Segmento ID4", lambda: (tello.move_forward(30), time.sleep(0.5)))

                if not ocupado() and trabajo["hilo"] is not None:
                    trabajo["hilo"] = None  
                    ref_linea = (w / 2, h * 0.75)
                    cambiar("SEGUIR LINEA", "Maniobra terminada, retomando pista")

            for t in tags:
                es_valido_por_area = area_tag(t) >= MIN_AREA_TAG or t.tag_id == 3
                color_tag = (0, 255, 0) if es_valido_por_area else (0, 0, 255)
                
                cv2.polylines(vista, [t.corners.astype(np.int32)], True, color_tag, 2)
                cv2.putText(vista, f"ID {t.tag_id}", tuple(t.corners[0].astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tag, 2)

            cv2.putText(vista, f"Etapa: {etapa} | FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vista, status_msg, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Dron Pista", vista)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  
                if not is_flying:
                    print("Despegando...")
                    tello.takeoff()
                    is_flying = True
                    cambiar("SEGUIR LINEA", "Iniciando recorrido")
            elif key == ord('q') or key == 27:  
                print("Saliendo por comando de usuario...")
                break

    except Exception as e:
        print(f"Error en bucle principal: {e}")

    finally:
        print("Finalizando y cerrando conexiones de forma segura...")
        if ocupado():
            try:
                trabajo["hilo"].join(timeout=3)
            except Exception:
                pass
        if is_flying:
            try:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
            except Exception:
                pass
        
        try:
            tello.streamoff()
        except Exception:
            pass
        try:
            tello.end()
        except Exception:
            pass
        
        cv2.destroyAllWindows()
        print("Programa cerrado correctamente.")

if __name__ == "__main__":
    main()