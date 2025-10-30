# Training script for LEAF-YOLO-Edge-Drone
# This script trains the model with FPN + CBAM/ECA + DeformableConv

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "LEAF-YOLO-Edge-Drone Training Script" -ForegroundColor Cyan
Write-Host "Architecture: FPN + CBAM/ECA + DeformableConv (DCN)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$WORKERS = 8
$DEVICE = "0"
$BATCH_SIZE = 16
$EPOCHS = 300
$IMG_SIZE = 640
$CFG = "cfg/LEAF-YOLO/leaf-edge-drone.yaml"
$DATA = "data/visdrone.yaml"
$HYP = "data/hyp.scratch.visdrone.yaml"
$NAME = "leaf-edge-drone"

Write-Host "Training Configuration:" -ForegroundColor Yellow
Write-Host "- Workers: $WORKERS"
Write-Host "- Device: $DEVICE"
Write-Host "- Batch Size: $BATCH_SIZE"
Write-Host "- Epochs: $EPOCHS"
Write-Host "- Image Size: $IMG_SIZE"
Write-Host "- Config: $CFG"
Write-Host "- Data: $DATA"
Write-Host "- Hyperparameters: $HYP"
Write-Host "- Run Name: $NAME"
Write-Host ""

Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

uv run train.py `
    --workers $WORKERS `
    --device $DEVICE `
    --batch-size $BATCH_SIZE `
    --epochs $EPOCHS `
    --data $DATA `
    --img $IMG_SIZE $IMG_SIZE `
    --cfg $CFG `
    --weights "" `
    --hyp $HYP `
    --name $NAME `
    --cache

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "Check results in: runs/train/$NAME/" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
