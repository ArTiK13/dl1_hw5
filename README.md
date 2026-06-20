# Lensless Computational Imaging

Код восстанавливает исходные измерения из безлинзовых измерений DigiCam с помощью фиксированного ADMM, обучаемого ADMM, модульных вариантов DRUNet + LeADMM и FISTA.

Лучшая итоговая модель по PSNR: `modular_pre_post` (`Pre4+LeADMM5+Post4`), PSNR на тесте `17.2964`.

## Установка

Для полной воспроизводимости после установки всех библиотек и проверки что все корректно работает текущее окружение с фиксацией всех версий (в т.ч. зависимостей) было записано в requirements.txt.
Установить его в чистый venv можно так:

```bash
pip install -r requirements.txt
```

Для запуска `demo.ipynb` в Colab проще установить уменьшенный набор библиотек:

```bash
pip install -q -r requirements_demo.txt
```

## Чекпоинты

```bash
python scripts/download_checkpoints.py --output-dir checkpoints
```

Репозиторий с обученными моделями: `xartik/hw5-lensless-computational-imaging-checkpoints`.

Пути скачанных чекпоинтов:

| Модель | Путь |
|---|---|
| `leadmm20` | `checkpoints/leadmm20.pth` |
| `modular_pre_post` | `checkpoints/modular_pre_post.pth` |
| `modular_pre_only` | `checkpoints/modular_pre_only.pth` |
| `modular_post_only` | `checkpoints/modular_post_only.pth` |

## Обучение

One-batch проверки перед полным обучением:

```bash
python train.py -cn leadmm20_onebatch
python train.py -cn modular_pre_post_onebatch
python train.py -cn modular_pre_only_onebatch
python train.py -cn modular_post_only_onebatch
```

Полные запуски обучаемых моделей:

```bash
python train.py -cn leadmm20
python train.py -cn modular_pre_post
python train.py -cn modular_pre_only
python train.py -cn modular_post_only
```

## Инференс на своих данных

Ожидаемая структура пользовательских данных:

```text
data_root/
  lensless/ImageID.png
  masks/ImageID.npy
  lensed/ImageID.png   # опционально
```

Запуск:

```bash
python inference.py -cn inference_custom \
  datasets.test.root_dir=/path/to/data_root \
  inferencer.save_path=custom_reconstructions \
  model=modular_pre_post \
  inferencer.from_pretrained=checkpoints/modular_pre_post.pth \
  inferencer.skip_model_load=false
```

Фиксированные методы не требуют чекпоинта:

```bash
python inference.py -cn inference_custom \
  datasets.test.root_dir=/path/to/data_root \
  inferencer.save_path=custom_admm100 \
  model=admm100 \
  inferencer.skip_model_load=true
```

Предсказания сохраняются в `data/saved/<save_path>/test/ImageID.png`.

## Метрики

```bash
python scripts/calculate_metrics.py \
  --gt-dir /path/to/data_root/lensed \
  --recon-dir data/saved/custom_reconstructions/test
```

Скрипт выводит PSNR, SSIM, MSE и LPIPS.

## Скорость

```bash
python scripts/benchmark_speed.py --device cuda --warmup 10 --trials 100
```

Основное значение: время работы модели в миллисекундах на изображение.

## Итоговые результаты

Все метрики ниже считаются на сохраненных выходах для `1500/1500` тестовых изображений DigiCam.

| Метод | PSNR | SSIM | MSE | LPIPS |
|---|---:|---:|---:|---:|
| ADMM100 | 11.9928 | 0.3493 | 0.076489 | 0.7808 |
| FISTA100 | 11.9588 | 0.2683 | 0.077316 | 0.7651 |
| LeADMM20 | 13.7775 | 0.3866 | 0.047398 | 0.7380 |
| Pre8+LeADMM5 | 14.9538 | 0.3162 | 0.036932 | 0.6984 |
| LeADMM5+Post8 | 16.7523 | 0.4468 | 0.026163 | 0.5842 |
| Pre4+LeADMM5+Post4 | 17.2964 | 0.4599 | 0.023356 | 0.5655 |

## Отчет

Итоговый отчет со сравнением всех методов находится в `analysis/analysis.ipynb`.
