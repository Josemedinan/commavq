# Student Long Final

## Cambio ganador

La mejora real no vino de cambiar arquitectura, sino de exponer el mismo `student` a mucha mas profundidad temporal del dataset real.

Hecho clave:

- los segmentos reales tienen `1200` frames
- los mejores checkpoints anteriores se entrenaban solo con `32` frames por segmento
- eso dejaba casi todo el dataset fuera del entrenamiento

Ruta final ejecutada:

1. `student_final_curriculum.pt`
2. fine-tune `128` frames -> `student_long128_ft.pt`
3. fine-tune `256` frames -> `student_long256_ft.pt`
4. fine-tune `512` frames -> `student_long512_ft.pt`
5. fine-tune full-length `1200` frames, 2 epocas mas -> `student_full1200_ft3.pt`

Artefacto final cuantizado:

- `artifacts/student_full1200_ft3_q8.bin`

## Calidad del modelo

Mejoras de validacion:

| checkpoint | val bpt |
| --- | ---: |
| `student_final_curriculum.pt` | `5.5711` |
| `student_long128_ft.pt` | `5.3310` |
| `student_long256_ft.pt` | `5.1278` |
| `student_long512_ft.pt` | `4.8785` |
| `student_full1200_ft.pt` | `4.3401` |
| `student_full1200_ft2.pt` | `4.2656` |
| `student_full1200_ft3.pt` | `4.2269` |

## Benchmarks reales

### 2 segmentos x 128 frames

| model | predicted bpt | effective bpt | archive bpt | ratio |
| --- | ---: | ---: | ---: | ---: |
| baseline current | `5.3150` | `5.6078` | `5.6382` | `1.7736x` |
| `long128_ft` | `4.9223` | `5.2396` | `5.2773` | `1.8949x` |
| `long256_ft` | `4.7466` | `5.0749` | `5.1133` | `1.9557x` |
| `long512_ft` | `4.5440` | `4.8850` | `4.9221` | `2.0316x` |

### 2 segmentos x 512 frames

| model | predicted bpt | effective bpt | archive bpt | ratio |
| --- | ---: | ---: | ---: | ---: |
| baseline current | `5.3408` | `5.4136` | `5.4302` | `1.8415x` |
| `long512_ft` | `4.5414` | `4.6266` | `4.6526` | `2.1493x` |

### 2 segmentos x 1200 frames

| model | predicted bpt | effective bpt | archive bpt | ratio |
| --- | ---: | ---: | ---: | ---: |
| baseline current | `5.5641` | `5.5937` | `5.6038` | `1.7845x` |
| `full1200_ft2` | `4.5186` | `4.5552` | `4.5765` | `2.1851x` |
| `full1200_ft3` | `4.4758` | `4.5126` | `4.5355` | `2.2048x` |

## Objetivo 2.2x

Objetivo:

- `2.2x` equivale a `4.5455 archive bpt`

Resultado local final:

- `4.5355 archive bpt`
- `2.2048x`

Eso cruza el objetivo localmente en el benchmark de `4 segmentos completos x 1200 frames`:

| subset | archive bpt | ratio |
| --- | ---: | ---: |
| `4 x 1200`, `student_full1200_ft3_q8.bin` | `4.5296` | `2.2077x` |

## Submission candidate

Archive construido:

- `artifacts/student_full1200_ft3_4seg.bin`

Zip autocontenido:

- `artifacts/student_full1200_ft3_submission.zip`

Tree standalone:

- `artifacts/student_full1200_ft3_tree`

Validacion standalone:

- `decompress.py` ejecutado contra el tree standalone: `ok`
- reconstruccion exacta en `4` segmentos: `true`

## Cambios de codigo relevantes

- `strong_compression/__init__.py`
  - se eliminaron imports pesados que estaban frenando scripts simples
- `strong_compression/student_quantization.py`
  - se redujo a la familia final `student_model.py`, eliminando carga de variantes experimentales
- `strong_compression/student_submission.py`
  - se redujo el package standalone a solo los modulos necesarios para la ruta final del student

## Conclusion

La ruta que funciono fue:

- mantener la arquitectura student actual
- dejar de subentrenarla con ventanas cortas
- entrenarla progresivamente sobre secuencias reales cada vez mas largas

Con eso, el student paso de un regime `~5.6 archive bpt` a uno `~4.53 archive bpt` en segmentos completos, suficiente para cruzar `2.2x` localmente sin GPT en decode.
