"""
Brain Cancer Segmentation Prediction System
COCO formatındaki annotation verilerini kullanarak beyin kanseri segmentasyon tahmini yapar
"""

import os
import json
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (512, 512)  # Reduced for faster inference
NUM_CLASSES = 2  # Background + Cancer

# GPU optimization settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_model(model_path):
    """Eğitilmiş modeli yükle"""
    
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
    
    # Checkpoint yükle
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model yüklendi: {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'val_iou' in checkpoint:
        print(f"  Val IoU: {checkpoint.get('val_iou'):.4f}")
    
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
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image


def predict(model, image_tensor):
    """Tahmin yap"""
    
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)['out']
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        prob_map = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
    
    return pred_mask, prob_map


def visualize(image, pred_mask, prob_map, save_path=None):
    """Gelişmiş görselleştirme - Beyin kanseri için özelleştirilmiş"""
    
    # 2 satır, 5 sütun = 10 farklı görselleştirme
    fig = plt.figure(figsize=(25, 10))
    
    # ===== ÜST SATIR =====
    
    # 1. Orijinal görüntü
    ax1 = plt.subplot(2, 5, 1)
    ax1.imshow(image)
    ax1.set_title('1. Orijinal Beyin Görüntüsü', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Binary mask
    ax2 = plt.subplot(2, 5, 2)
    ax2.imshow(pred_mask, cmap='gray')
    ax2.set_title('2. Kanser Bölgesi Maskesi', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Probability map (Jet colormap)
    ax3 = plt.subplot(2, 5, 3)
    im1 = ax3.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    ax3.set_title('3. Kanser Olasılığı (Jet)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im1, ax=ax3, fraction=0.046)
    
    # 4. Probability map (Hot colormap)
    ax4 = plt.subplot(2, 5, 4)
    im2 = ax4.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    ax4.set_title('4. Kanser Olasılığı (Hot)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im2, ax=ax4, fraction=0.046)
    
    # 5. Contour (kontur çizimi)
    ax5 = plt.subplot(2, 5, 5)
    ax5.imshow(image)
    contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if contour.shape[0] > 5:  # En az 5 nokta
            contour = contour.squeeze()
            if len(contour.shape) == 2:
                ax5.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
    ax5.set_title('5. Kanser Kontur Çizimi', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # ===== ALT SATIR =====
    
    # 6. Kırmızı overlay
    ax6 = plt.subplot(2, 5, 6)
    overlay_red = np.array(image).copy()
    mask_colored = np.zeros_like(overlay_red)
    mask_colored[pred_mask == 1] = [255, 0, 0]
    overlay_red = cv2.addWeighted(overlay_red, 0.7, mask_colored, 0.3, 0)
    ax6.imshow(overlay_red)
    ax6.set_title('6. Kanser Overlay (Kırmızı)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 7. Yeşil overlay
    ax7 = plt.subplot(2, 5, 7)
    overlay_green = np.array(image).copy()
    mask_colored_green = np.zeros_like(overlay_green)
    mask_colored_green[pred_mask == 1] = [0, 255, 0]
    overlay_green = cv2.addWeighted(overlay_green, 0.7, mask_colored_green, 0.3, 0)
    ax7.imshow(overlay_green)
    ax7.set_title('7. Kanser Overlay (Yeşil)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # 8. Güven skoru threshold (0.5)
    ax8 = plt.subplot(2, 5, 8)
    high_conf_mask = (prob_map > 0.5).astype(np.uint8)
    ax8.imshow(high_conf_mask, cmap='Reds')
    cancer_pixels = np.sum(pred_mask == 1)
    high_conf_pixels = np.sum(high_conf_mask == 1)
    ax8.set_title(f'8. Yüksek Güven (>0.5)\n{high_conf_pixels}/{cancer_pixels} piksel',
                  fontsize=11, fontweight='bold')
    ax8.axis('off')
    
    # 9. Kenar tespiti
    ax9 = plt.subplot(2, 5, 9)
    edges = cv2.Canny((pred_mask * 255).astype(np.uint8), 50, 150)
    overlay_edges = np.array(image).copy()
    overlay_edges[edges > 0] = [255, 255, 0]  # Sarı
    ax9.imshow(overlay_edges)
    ax9.set_title('9. Kanser Kenar Tespiti', fontsize=12, fontweight='bold')
    ax9.axis('off')
    
    # 10. Bölge analizi (alan bilgisi)
    ax10 = plt.subplot(2, 5, 10)
    ax10.imshow(image)
    
    # Her ölgeyi farklı renkle işaretle
    labeled_mask = cv2.connectedComponents(pred_mask.astype(np.uint8))[1]
    num_regions = np.max(labeled_mask)
    
    if num_regions > 0:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_regions))
        for i in range(1, num_regions + 1):
            region_mask = (labeled_mask == i)
            region_area = np.sum(region_mask)
            
            # Bölge merkezini bul
            y_coords, x_coords = np.where(region_mask)
            if len(y_coords) > 0:
                center_y, center_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                
                # Renkli nokta
                color_idx = (i - 1) % len(colors)
                ax10.scatter(center_x, center_y, c=[colors[color_idx]], s=200, marker='o',
                           edgecolors='white', linewidths=2)
                
                # Alan bilgisi
                ax10.text(center_x, center_y, f'{region_area}px',
                         ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax10.set_title(f'10. Kanser Bölge Analizi ({num_regions} bölge)', fontsize=12, fontweight='bold')
    ax10.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Görsel kaydedildi: {save_path}")
    
    plt.show()
    
    # Bölge istatistikleri yazdır
    if num_regions > 0:
        print(f"\n📊 Kanser Bölge İstatistikleri:")
        print(f"  Toplam Kanser Bölge Sayısı: {num_regions}")
        for i in range(1, min(num_regions + 1, 11)):  # İlk 10 bölge
            region_area = np.sum(labeled_mask == i)
            print(f"  Kanser Bölge {i}: {region_area} piksel")


def calculate_stats(pred_mask, prob_map):
    """İstatistikleri hesapla"""
    
    total_pixels = pred_mask.size
    cancer_pixels = np.sum(pred_mask == 1)
    cancer_percentage = (cancer_pixels / total_pixels) * 100
    avg_confidence = np.mean(prob_map[pred_mask == 1]) if cancer_pixels > 0 else 0
    
    return {
        'total_pixels': total_pixels,
        'cancer_pixels': cancer_pixels,
        'cancer_percentage': cancer_percentage,
        'avg_confidence': avg_confidence
    }


def print_stats(stats):
    """İstatistikleri yazdır"""
    
    print(f"\n{'='*60}")
    print("BEYİN KANSERİ TAHMİN İSTATİSTİKLERİ")
    print(f"{'='*60}")
    print(f"Toplam Piksel        : {stats['total_pixels']:,}")
    print(f"Kanser Piksel        : {stats['cancer_pixels']:,}")
    print(f"Kanser Yüzdesi       : {stats['cancer_percentage']:.2f}%")
    print(f"Ortalama Güven       : {stats['avg_confidence']:.4f}")
    print(f"{'='*60}\n")


def predict_single(model_path, image_path, output_dir='results'):
    """
    Tek görüntü üzerinde tahmin yap
    
    Args:
        model_path: Model dosyası yolu
        image_path: Görüntü dosyası yolu
        output_dir: Çıktı klasörü
    
    Returns:
        pred_mask, prob_map, stats
    """
    
    print(f"\n{'='*60}")
    print(f"Beyin Görüntüsü: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Model yükle
    model = load_model(model_path)
    
    # Görüntü hazırla
    image_tensor, image = preprocess_image(image_path)
    
    # Tahmin
    pred_mask, prob_map = predict(model, image_tensor)
    
    # İstatistikler
    stats = calculate_stats(pred_mask, prob_map)
    print_stats(stats)
    
    # Görselleştir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f'{base_name}_cancer_result_{timestamp}.png')
    
    visualize(image, pred_mask, prob_map, save_path)
    
    # Mask kaydet
    mask_path = os.path.join(output_dir, f'{base_name}_cancer_mask_{timestamp}.png')
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_path)
    print(f"✓ Kanser maskesi kaydedildi: {mask_path}")
    
    return pred_mask, prob_map, stats


def predict_folder(model_path, folder_path, output_dir='results', limit=None):
    """
    Klasördeki tüm görüntüleri işle
    
    Args:
        model_path: Model dosyası yolu
        folder_path: Görüntü klasörü
        output_dir: Çıktı klasörü
        limit: Maksimum işlenecek görüntü sayısı (None = hepsi)
    
    Returns:
        results: Her görüntü için istatistikler listesi
    """
    
    print(f"\n{'='*60}")
    print(f"Beyin Görüntüleri Klasörü: {folder_path}")
    print(f"{'='*60}")
    
    # Model yükle
    model = load_model(model_path)
    
    # Görüntü dosyalarını bul
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in extensions
    ]
    
    if limit:
        image_files = image_files[:limit]
    
    print(f"\n{len(image_files)} beyin görüntüsü bulundu.\n")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {os.path.basename(image_path)}")
        
        try:
            # Görüntü hazırla
            image_tensor, image = preprocess_image(image_path)
            
            # Tahmin
            pred_mask, prob_map = predict(model, image_tensor)
            
            # İstatistikler
            stats = calculate_stats(pred_mask, prob_map)
            stats['filename'] = os.path.basename(image_path)
            results.append(stats)
            
            print(f"  Kanser Oranı: {stats['cancer_percentage']:.2f}%")
            
            # Mask kaydet
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(output_dir, f'{base_name}_cancer_mask_{timestamp}.png')
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_path)
            
        except Exception as e:
            print(f"  ✗ Hata: {e}")
    
    # Özet istatistikler
    if results:
        print(f"\n{'='*60}")
        print("ÖZET İSTATİSTİKLER")
        print(f"{'='*60}")
        avg_cancer = np.mean([r['cancer_percentage'] for r in results])
        avg_conf = np.mean([r['avg_confidence'] for r in results])
        print(f"Ortalama Kanser Yüzdesi: {avg_cancer:.2f}%")
        print(f"Ortalama Güven Skoru    : {avg_conf:.4f}")
        print(f"İşlenen Görüntü Sayısı  : {len(results)}")
        print(f"{'='*60}\n")
    
    return results


def predict_from_train_folder(model_path, output_dir='results', limit=10):
    """
    Train klasöründeki görüntüleri tahmin et
    
    Args:
        model_path: Model dosyası yolu
        output_dir: Çıktı klasörü
        limit: Maksimum işlenecek görüntü sayısı
    
    Returns:
        results: Her görüntü için istatistikler listesi
    """
    
    train_folder = 'train'
    
    if not os.path.exists(train_folder):
        print(f"❌ Train klasörü bulunamadı: {train_folder}")
        return []
    
    print(f"\n{'='*60}")
    print("BEYİN KANSERİ SEGMENTASYON TAHMİNİ")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Train Klasörü: {train_folder}")
    print(f"Çıktı: {output_dir}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")
    
    return predict_folder(model_path, train_folder, output_dir, limit)


# Kolay kullanım için kısa isimler
predict_image = predict_single
predict_images = predict_folder


def demo(image_path=None, model_path='models/best_model.pth'):
    """
    Hızlı demo - varsayılan ayarlarla tahmin yap
    
    Kullanım:
        brain_cancer_prediction.demo('train/103_jpg.rf.545b7cac5ee5582ca96be24af6e900fe.jpg')
    """
    
    if image_path is None:
        print("❌ Görüntü yolu gerekli!")
        print("\nÖrnek kullanım:")
        print("  brain_cancer_prediction.demo('train/103_jpg.rf.545b7cac5ee5582ca96be24af6e900fe.jpg')")
        return
    
    return predict_single(
        model_path=model_path,
        image_path=image_path,
        output_dir='results'
    )


def quick_predict(image_path, model_path='models/best_model.pth'):
    """
    Hızlı tahmin - sadece sonuçları döndür, kaydetme
    
    Kullanım:
        mask, prob, stats = brain_cancer_prediction.quick_predict('train/image.jpg')
    """
    
    model = load_model(model_path)
    image_tensor, image = preprocess_image(image_path)
    pred_mask, prob_map = predict(model, image_tensor)
    stats = calculate_stats(pred_mask, prob_map)
    
    print_stats(stats)
    visualize(image, pred_mask, prob_map, save_path=None)
    
    return pred_mask, prob_map, stats


if __name__ == '__main__':
    """
    Direkt çalıştırıldığında test yap
    """
    
    # Varsayılan ayarlar
    MODEL_PATH = 'models/best_model.pth'
    OUTPUT_DIR = 'results'
    
    print("="*60)
    print("BEYİN KANSERİ SEGMENTASYON TAHMİNİ")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Train klasöründeki görüntüleri test et
    print("\n📸 TRAIN KLASÖRÜ TESTİ")
    results = predict_from_train_folder(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        limit=5
    )
    
    if results:
        print("\n✅ Test tamamlandı!")
        print(f"İşlenen görüntü sayısı: {len(results)}")
    else:
        print("\n⚠️ Test görüntüleri bulunamadı!")
    
    print("\n" + "="*60)
    print("TEST TAMAMLANDI!")
    print("="*60)
