"""
Tooth Decay Segmentation Prediction Script
Processes images from target/data and saves predictions to target/pred
"""

import os
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (512, 512)
NUM_CLASSES = 5  # Background + Dolgu + Kanal + Çürük + Protez

# Paths
INPUT_DIR = 'target/data'
OUTPUT_DIR = 'target/pred'
MODEL_DIR = 'models'

# Kategori renkleri (RGB)
CATEGORY_COLORS = {
    0: [0, 0, 0],        # Background - Siyah
    1: [255, 0, 0],      # Dolgu - Kırmızı
    2: [0, 255, 0],      # Kanal - Yeşil
    3: [0, 0, 255],      # Çürük - Mavi
    4: [255, 255, 0]     # Protez - Sarı
}

# Kategori isimleri
CATEGORY_NAMES = {
    0: 'Background',
    1: 'Dolgu',
    2: 'Kanal',
    3: 'Çürük',
    4: 'Protez'
}


def load_model(model_path):
    """Eğitilmiş modeli yükle"""
    
    # Dosya kontrolü
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        raise ValueError(f"Model dosyası boş: {model_path}")
    
    print(f"Model dosyası boyutu: {file_size / (1024*1024):.2f} MB")
    
    # Model oluştur
    try:
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    except ImportError:
        model = deeplabv3_resnet50(pretrained=True)
    
    # Son katmanı değiştir
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    
    # aux_classifier varsa onu da değiştir
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    
    # Checkpoint yükle - farklı yöntemler dene
    try:
        # Önce weights_only=False ile dene
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except Exception as e1:
        print(f"İlk yükleme denemesi başarısız: {e1}")
        try:
            # weights_only parametresi olmadan dene
            checkpoint = torch.load(model_path, map_location=DEVICE)
        except Exception as e2:
            print(f"İkinci yükleme denemesi başarısız: {e2}")
            try:
                # pickle_module ile dene
                import pickle
                checkpoint = torch.load(model_path, map_location=DEVICE, pickle_module=pickle)
            except Exception as e3:
                raise RuntimeError(
                    f"Model dosyası yüklenemedi. Dosya bozuk olabilir.\n"
                    f"Hatalar:\n  1. {e1}\n  2. {e2}\n  3. {e3}\n"
                    f"Dosya yolu: {model_path}\n"
                    f"Dosya boyutu: {file_size} bytes"
                )
    
    # State dict yükle
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Eğer direkt state_dict ise
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model yüklendi: {model_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_iou' in checkpoint:
        print(f"  Val IoU: {checkpoint['val_iou']:.4f}")
    
    return model


def preprocess_image(image_path):
    """Görüntüyü model için hazırla"""
    
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image, original_size


def predict(model, image_tensor):
    """Tahmin yap"""
    
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)['out']
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        prob_maps = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    
    return pred_mask, prob_maps


def create_colored_mask(pred_mask):
    """Renkli segmentasyon maskesi oluştur"""
    
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for cls_id, color in CATEGORY_COLORS.items():
        colored_mask[pred_mask == cls_id] = color
    
    return colored_mask


def resize_mask_to_original(mask, original_size):
    """Mask'i orijinal görüntü boyutuna resize et"""
    
    mask_pil = Image.fromarray(mask)
    mask_resized = mask_pil.resize(original_size, Image.NEAREST)
    return np.array(mask_resized)


def save_predictions(image_name, original_image, pred_mask, prob_maps, output_dir):
    """Tahmin sonuçlarını tek bir composite görüntüde kaydet"""
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]
    
    # Görüntüleri hazırla
    original_resized = np.array(original_image.resize(IMG_SIZE))
    colored_mask = create_colored_mask(pred_mask)
    overlay = (original_resized * 0.6 + colored_mask * 0.4).astype(np.uint8)
    
    # Orijinal boyutta
    original_size = original_image.size
    original_array = np.array(original_image)
    colored_mask_original = resize_mask_to_original(colored_mask, original_size)
    overlay_original = (original_array * 0.6 + colored_mask_original * 0.4).astype(np.uint8)
    
    # Olasılık haritalarını hazırla
    prob_images = {}
    for cls_id in range(1, NUM_CLASSES):
        prob_map = prob_maps[cls_id]
        prob_map_uint8 = (prob_map * 255).astype(np.uint8)
        # Grayscale'i RGB'ye çevir (colormap için)
        prob_images[cls_id] = prob_map_uint8
    
    # Composite görüntü oluştur: 3 satır, 4 sütun
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Segmentasyon Sonuçları: {image_name}', fontsize=16, fontweight='bold')
    
    # İlk satır: Orijinal boyutlu görüntüler
    axes[0, 0].imshow(original_array)
    axes[0, 0].set_title('1. Orijinal Görüntü\n(Orijinal Boyut)', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(colored_mask_original)
    axes[0, 1].set_title('2. Segmentasyon Maskesi\n(Orijinal Boyut)', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(overlay_original)
    axes[0, 2].set_title('3. Overlay\n(Orijinal Boyut)', fontsize=10, fontweight='bold')
    axes[0, 2].axis('off')
    
    # İstatistikler
    stats = calculate_stats(pred_mask, prob_maps)
    stats_text = "Kategori Dağılımı:\n"
    for category, data in stats.items():
        stats_text += f"{category:10s}: {data['percentage']:5.2f}%\n"
        stats_text += f"  ({data['count']:,} piksel)\n"
        stats_text += f"  Güven: {data['avg_confidence']:.4f}\n\n"
    
    axes[0, 3].text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                    verticalalignment='center', transform=axes[0, 3].transAxes)
    axes[0, 3].set_title('4. İstatistikler', fontsize=10, fontweight='bold')
    axes[0, 3].axis('off')
    
    # İkinci satır: Resize edilmiş görüntüler
    axes[1, 0].imshow(original_resized)
    axes[1, 0].set_title('5. Orijinal Görüntü\n(512x512)', fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(colored_mask)
    axes[1, 1].set_title('6. Segmentasyon Maskesi\n(512x512)', fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('7. Overlay\n(512x512)', fontsize=10, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Renk legend
    legend_text = "Kategori Renkleri:\n"
    for cls_id in range(1, NUM_CLASSES):
        color = tuple(c / 255.0 for c in CATEGORY_COLORS[cls_id])
        legend_text += f"  {CATEGORY_NAMES[cls_id]}\n"
        axes[1, 3].add_patch(Rectangle((0.1, 0.7 - cls_id*0.15), 0.1, 0.1, 
                                      facecolor=color, edgecolor='black', linewidth=1,
                                      transform=axes[1, 3].transAxes))
        axes[1, 3].text(0.25, 0.75 - cls_id*0.15, CATEGORY_NAMES[cls_id], 
                        fontsize=10, transform=axes[1, 3].transAxes, 
                        verticalalignment='center')
    axes[1, 3].set_title('8. Renk Legend', fontsize=10, fontweight='bold')
    axes[1, 3].axis('off')
    
    # Üçüncü satır: Olasılık haritaları
    colormaps = ['Reds', 'Greens', 'Blues', 'YlOrRd']
    for idx, cls_id in enumerate(range(1, NUM_CLASSES)):
        im = axes[2, idx].imshow(prob_images[cls_id], cmap=colormaps[idx], vmin=0, vmax=255)
        axes[2, idx].set_title(f'{9+idx}. {CATEGORY_NAMES[cls_id]} Olasılığı', 
                              fontsize=10, fontweight='bold')
        axes[2, idx].axis('off')
        plt.colorbar(im, ax=axes[2, idx], fraction=0.046)
    
    # Son sütun boş kalırsa, kategori dağılım grafiği ekle
    if NUM_CLASSES - 1 <= 3:  # 4 kategori varsa (NUM_CLASSES=5, background hariç)
        category_counts = {}
        for cls_id in range(1, NUM_CLASSES):
            count = np.sum(pred_mask == cls_id)
            category_counts[CATEGORY_NAMES[cls_id]] = count
        
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        colors_list = [tuple(c / 255.0 for c in CATEGORY_COLORS[i+1]) for i in range(len(categories))]
        
        axes[2, 3].bar(categories, counts, color=colors_list)
        axes[2, 3].set_title('12. Kategori Dağılımı', fontsize=10, fontweight='bold')
        axes[2, 3].set_ylabel('Piksel Sayısı')
        axes[2, 3].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Kaydet
    composite_path = os.path.join(output_dir, f'{base_name}_composite.png')
    plt.savefig(composite_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Composite görüntü kaydedildi: {composite_path}")
    
    return {
        'composite': composite_path
    }


def calculate_stats(pred_mask, prob_maps):
    """İstatistikleri hesapla"""
    
    total_pixels = pred_mask.size
    stats = {}
    
    for cls_id in range(1, NUM_CLASSES):
        category_name = CATEGORY_NAMES[cls_id]
        count = np.sum(pred_mask == cls_id)
        percentage = (count / total_pixels) * 100
        avg_conf = np.mean(prob_maps[cls_id][pred_mask == cls_id]) if count > 0 else 0
        
        stats[category_name] = {
            'count': count,
            'percentage': percentage,
            'avg_confidence': avg_conf
        }
    
    return stats


def process_single_image(model, image_path, output_dir):
    """Tek bir görüntüyü işle"""
    
    image_name = os.path.basename(image_path)
    print(f"\nİşleniyor: {image_name}")
    
    # Preprocess
    image_tensor, original_image, original_size = preprocess_image(image_path)
    
    # Predict
    pred_mask, prob_maps = predict(model, image_tensor)
    
    # Statistics
    stats = calculate_stats(pred_mask, prob_maps)
    
    # Save predictions
    saved_paths = save_predictions(image_name, original_image, pred_mask, prob_maps, output_dir)
    
    # Print stats
    print(f"  Kategori Dağılımı:")
    for category, data in stats.items():
        print(f"    {category:10s}: {data['percentage']:5.2f}% ({data['count']:8,} piksel) - Güven: {data['avg_confidence']:.4f}")
    
    return stats, saved_paths


def find_best_model(model_dir):
    """En iyi modeli bul (best_model.pth veya en büyük/en yüksek epoch checkpoint)"""
    
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        return best_model_path
    
    # Checkpoint'leri kontrol et
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if checkpoints:
        # Dosya boyutlarına göre sırala (büyük dosyalar daha tam olabilir)
        checkpoint_paths = [os.path.join(model_dir, f) for f in checkpoints]
        checkpoint_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
        
        # Önce en büyük dosyayı dene
        for checkpoint_path in checkpoint_paths:
            file_size = os.path.getsize(checkpoint_path) / (1024*1024)  # MB
            print(f"Model bulundu: {os.path.basename(checkpoint_path)} ({file_size:.2f} MB)")
            return checkpoint_path
        
        # Eğer hiçbiri çalışmazsa, en yüksek epoch'u dene
        epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
        max_epoch = max(epochs)
        checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{max_epoch}.pth')
        return checkpoint_path
    
    raise FileNotFoundError(f"Model bulunamadı: {model_dir}")


def main():
    """Ana fonksiyon"""
    
    print("="*60)
    print("Tooth Decay Segmentasyon Tahmini")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print("="*60)
    
    # Model yükle
    try:
        model_path = find_best_model(MODEL_DIR)
        model = load_model(model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ Hata: {e}")
        print("\nMevcut modeller:")
        if os.path.exists(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                if f.endswith('.pth'):
                    file_path = os.path.join(MODEL_DIR, f)
                    file_size = os.path.getsize(file_path) / (1024*1024)
                    print(f"  - {f} ({file_size:.2f} MB)")
        
        # Alternatif modelleri dene
        print("\nAlternatif modeller deneniyor...")
        model = None
        if os.path.exists(MODEL_DIR):
            checkpoints = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
            tried_path = model_path if 'model_path' in locals() else None
            for checkpoint in sorted(checkpoints, reverse=True):
                if tried_path and checkpoint == os.path.basename(tried_path):
                    continue
                try:
                    alt_path = os.path.join(MODEL_DIR, checkpoint)
                    print(f"\nAlternatif model deneniyor: {checkpoint}")
                    model = load_model(alt_path)
                    print(f"✓ Alternatif model başarıyla yüklendi: {checkpoint}")
                    break
                except Exception as e2:
                    print(f"  ✗ Başarısız: {e2}")
                    continue
        
        if model is None:
            print("\n❌ Hiçbir model yüklenemedi!")
            return
    
    # Görüntü dosyalarını bul
    if not os.path.exists(INPUT_DIR):
        print(f"\n❌ Hata: Input dizini bulunamadı: {INPUT_DIR}")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if not image_files:
        print(f"\n❌ Hata: {INPUT_DIR} dizininde görüntü bulunamadı")
        return
    
    print(f"\n{len(image_files)} görüntü bulundu.\n")
    
    # Tüm görüntüleri işle
    all_stats = []
    for image_path in tqdm(image_files, desc="İşleniyor"):
        try:
            stats, saved_paths = process_single_image(model, image_path, OUTPUT_DIR)
            stats['filename'] = os.path.basename(image_path)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n❌ Hata ({os.path.basename(image_path)}): {e}")
    
    # Özet istatistikler
    if all_stats:
        print(f"\n{'='*60}")
        print("ÖZET İSTATİSTİKLER")
        print(f"{'='*60}")
        for category_name in CATEGORY_NAMES.values():
            if category_name == 'Background':
                continue
            percentages = [s[category_name]['percentage'] for s in all_stats]
            avg_percentage = np.mean(percentages)
            print(f"Ortalama {category_name:10s} Yüzdesi: {avg_percentage:.2f}%")
        print(f"İşlenen Görüntü Sayısı: {len(all_stats)}")
        print(f"{'='*60}\n")
    
    print(f"✓ Tüm tahminler kaydedildi: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

