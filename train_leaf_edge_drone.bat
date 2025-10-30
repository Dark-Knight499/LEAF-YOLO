@echo off
REM Training script for LEAF-YOLO-Edge-Drone
REM This script trains the model with FPN + CBAM/ECA + DeformableConv

echo ============================================================
echo LEAF-YOLO-Edge-Drone Training Script
echo Architecture: FPN + CBAM/ECA + DeformableConv (DCN)
echo ============================================================
echo.

REM Configuration
set WORKERS=8
set DEVICE=0
set BATCH_SIZE=16
set EPOCHS=300
set IMG_SIZE=640
set CFG=cfg/LEAF-YOLO/leaf-edge-drone.yaml
set DATA=data/visdrone.yaml
set HYP=data/hyp.scratch.visdrone.yaml
set NAME=leaf-edge-drone
set CACHE=--cache

echo Training Configuration:
echo - Workers: %WORKERS%
echo - Device: %DEVICE%
echo - Batch Size: %BATCH_SIZE%
echo - Epochs: %EPOCHS%
echo - Image Size: %IMG_SIZE%
echo - Config: %CFG%
echo - Data: %DATA%
echo - Hyperparameters: %HYP%
echo - Run Name: %NAME%
echo.

echo Starting training...
echo.

uv run train.py ^
    --workers %WORKERS% ^
    --device %DEVICE% ^
    --batch-size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --data %DATA% ^
    --img %IMG_SIZE% %IMG_SIZE% ^
    --cfg %CFG% ^
    --weights "" ^
    --hyp %HYP% ^
    --name %NAME% ^
    %CACHE%

echo.
echo ============================================================
echo Training completed!
echo Check results in: runs/train/%NAME%/
echo ============================================================

pause
