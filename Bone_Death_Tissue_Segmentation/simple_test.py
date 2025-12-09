import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from scipy import ndimage

def calculate_area_mm2(pred_mask, prob_map, image_size=(512, 512), real_size_mm=(168.96, 168.96)):
    """Segmentasyon alanını mm² cinsinden hesapla"""
    
    print("\n" + "="*60)
    print("ALAN HESAPLAMASI")
    print("="*60)
    
    # Görüntü boyutları
    img_height, img_width = image_size
    real_height_mm, real_width_mm = real_size_mm
    
    # Piksel başına mm hesapla
    mm_per_pixel_height = real_height_mm / img_height
    mm_per_pixel_width = real_width_mm / img_width
    
    print(f"Goruntu boyutu: {img_width}x{img_height} piksel")
    print(f"Gercek boyut: {real_width_mm}x{real_height_mm} mm")
    print(f"Piksel basina mm: {mm_per_pixel_width:.4f} x {mm_per_pixel_height:.4f} mm")
    
    # Bir pikselin alanı (mm²)
    pixel_area_mm2 = mm_per_pixel_width * mm_per_pixel_height
    print(f"Bir pikselin alani: {pixel_area_mm2:.6f} mm²")
    
    # Segmentasyon alanı hesaplama
    bone_pixels = np.sum(pred_mask == 1)
    total_pixels = pred_mask.size
    bone_percentage = (bone_pixels / total_pixels) * 100
    
    # Toplam alan (mm²)
    total_area_mm2 = total_pixels * pixel_area_mm2
    bone_area_mm2 = bone_pixels * pixel_area_mm2
    
    print(f"\nSEGMENTASYON SONUCLARI:")
    print(f"Toplam piksel sayisi: {total_pixels:,}")
    print(f"Kemik piksel sayisi: {bone_pixels:,}")
    print(f"Kemik yuzdesi: {bone_percentage:.2f}%")
    print(f"Toplam alan: {total_area_mm2:.2f} mm²")
    print(f"Kemik alani: {bone_area_mm2:.2f} mm²")
    
    # Güven skoru analizi
    if bone_pixels > 0:
        avg_confidence = np.mean(prob_map[pred_mask == 1])
        max_confidence = np.max(prob_map[pred_mask == 1])
        min_confidence = np.min(prob_map[pred_mask == 1])
        
        print(f"\nGUVEN SKORU ANALIZI:")
        print(f"Ortalama guven: {avg_confidence:.4f}")
        print(f"En yuksek guven: {max_confidence:.4f}")
        print(f"En dusuk guven: {min_confidence:.4f}")
        
        # Farklı güven eşikleri ile alan hesaplama
        print(f"\nFARKLI GUVEN ESIKLERI ILE ALAN:")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for threshold in thresholds:
            high_conf_pixels = np.sum((pred_mask == 1) & (prob_map > threshold))
            high_conf_area = high_conf_pixels * pixel_area_mm2
            print(f"  >{threshold:.1f} guven: {high_conf_pixels:,} piksel = {high_conf_area:.2f} mm²")
    
    # Bölge analizi (connected components)
    labeled_mask, num_regions = ndimage.label(pred_mask == 1)
    
    if num_regions > 0:
        print(f"\nBOLGE ANALIZI:")
        print(f"Toplam bolge sayisi: {num_regions}")
        
        region_areas = []
        for i in range(1, num_regions + 1):
            region_pixels = np.sum(labeled_mask == i)
            region_area = region_pixels * pixel_area_mm2
            region_areas.append(region_area)
            print(f"  Bolge {i}: {region_pixels:,} piksel = {region_area:.2f} mm²")
        
        if region_areas:
            print(f"\nBOLGE ISTATISTIKLERI:")
            print(f"En buyuk bolge: {max(region_areas):.2f} mm²")
            print(f"En kucuk bolge: {min(region_areas):.2f} mm²")
            print(f"Ortalama bolge alani: {np.mean(region_areas):.2f} mm²")
    
    print("="*60)
    
    return {
        'bone_area_mm2': bone_area_mm2,
        'total_area_mm2': total_area_mm2,
        'bone_percentage': bone_percentage,
        'bone_pixels': bone_pixels,
        'pixel_area_mm2': pixel_area_mm2
    }

def create_detailed_visualization(image, pred_mask, prob_map, area_results, image_path):
    """Detaylı görselleştirme - alan bilgisi ile (PIL/OpenCV kullanarak)"""
    
    # Görüntüyü numpy array'e çevir
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # 1. Orijinal + Overlay
    overlay = image_np.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[pred_mask == 1] = [255, 0, 0]  # Kırmızı
    overlay_result = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    # 2. Binary Mask (renkli)
    binary_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    binary_colored[pred_mask == 1] = [255, 255, 255]  # Beyaz
    
    # 3. Olasılık Haritası (jet colormap benzeri)
    prob_colored = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 4. Kontur Çizimi
    contour_image = image_np.copy()
    contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)
    
    # 5. Bölge Analizi
    labeled_mask, num_regions = ndimage.label(pred_mask == 1)
    region_image = image_np.copy()
    
    if num_regions > 0:
        # Her bölge için farklı renk
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i in range(1, min(num_regions + 1, 7)):  # Maksimum 6 bölge
            region_mask = (labeled_mask == i)
            region_pixels = np.sum(region_mask)
            region_area = region_pixels * area_results['pixel_area_mm2']
            
            # Bölge merkezini bul ve işaretle
            y_coords, x_coords = np.where(region_mask)
            if len(y_coords) > 0:
                center_y, center_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                color = colors[(i - 1) % len(colors)]
                cv2.circle(region_image, (center_x, center_y), 10, color, -1)
                cv2.circle(region_image, (center_x, center_y), 10, (255, 255, 255), 2)
                cv2.putText(region_image, f'{region_area:.1f}mm²', 
                           (center_x - 30, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 6. İstatistikler görüntüsü
    stats_image = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Açık gri arka plan
    
    bone_area = area_results['bone_area_mm2']
    bone_pixels = area_results['bone_pixels']
    
    stats_text = [
        "ALAN ISTATISTIKLERI",
        "==================",
        "Goruntu Boyutu: 512x512 piksel",
        "Gercek Boyut: 168.96x168.96 mm",
        f"Piksel Alani: {area_results['pixel_area_mm2']:.6f} mm²",
        "",
        "KEMIK SEGMENTASYON",
        "==================",
        f"Kemik Alani: {bone_area:.2f} mm²",
        f"Kemik Piksel: {bone_pixels:,}",
        f"Kemik Yuzdesi: {area_results['bone_percentage']:.2f}%",
        f"Toplam Alan: {area_results['total_area_mm2']:.0f} mm²",
        "",
        "GUVEN SKORU",
        "===========",
        f"Ortalama: {np.mean(prob_map[pred_mask == 1]):.4f}",
        f"En Yuksek: {np.max(prob_map[pred_mask == 1]):.4f}",
        f"En Dusuk: {np.min(prob_map[pred_mask == 1]):.4f}"
    ]
    
    y_offset = 30
    for line in stats_text:
        cv2.putText(stats_image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += 25
    
    # Tüm görüntüleri birleştir (2x3 grid)
    # Her görüntüyü aynı boyuta getir
    target_size = (400, 400)
    
    overlay_resized = cv2.resize(overlay_result, target_size)
    binary_resized = cv2.resize(binary_colored, target_size)
    prob_resized = cv2.resize(prob_colored, target_size)
    contour_resized = cv2.resize(contour_image, target_size)
    region_resized = cv2.resize(region_image, target_size)
    stats_resized = cv2.resize(stats_image, target_size)
    
    # Üst satır
    top_row = np.hstack([overlay_resized, binary_resized, prob_resized])
    # Alt satır
    bottom_row = np.hstack([contour_resized, region_resized, stats_resized])
    
    # Tüm görüntüyü birleştir
    final_image = np.vstack([top_row, bottom_row])
    
    # Başlık ekle
    title_height = 50
    title_image = np.ones((title_height, final_image.shape[1], 3), dtype=np.uint8) * 50
    cv2.putText(title_image, f"Detayli Kemik Segmentasyon Analizi - {os.path.basename(image_path)}", 
               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Başlık ve içeriği birleştir
    final_with_title = np.vstack([title_image, final_image])
    
    # Kaydet
    cv2.imwrite('detailed_analysis.png', cv2.cvtColor(final_with_title, cv2.COLOR_RGB2BGR))
    print("Detayli analiz kaydedildi: detailed_analysis.png")
    
    # Görüntüyü göster
    cv2.imshow('Detayli Kemik Segmentasyon Analizi', final_with_title)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_simple_visualization(image, pred_mask, prob_map, area_results, test_image):
    """Basit görselleştirme (PIL/OpenCV kullanarak)"""
    
    # Görüntüyü numpy array'e çevir
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # 1. Orijinal görüntü
    original_resized = cv2.resize(image_np, (400, 400))
    
    # 2. Binary mask (renkli)
    binary_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    binary_colored[pred_mask == 1] = [255, 255, 255]  # Beyaz
    binary_resized = cv2.resize(binary_colored, (400, 400))
    
    # 3. Olasılık haritası
    prob_colored = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    prob_resized = cv2.resize(prob_colored, (400, 400))
    
    # Görüntüleri yan yana birleştir
    combined_image = np.hstack([original_resized, binary_resized, prob_resized])
    
    # Başlık ekle
    title_height = 60
    title_image = np.ones((title_height, combined_image.shape[1], 3), dtype=np.uint8) * 50
    
    # Başlık metni
    cv2.putText(title_image, f"Kemik Segmentasyon Testi - {os.path.basename(test_image)}", 
               (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Alt başlıklar
    cv2.putText(title_image, "Orijinal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(title_image, "Tahmin", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(title_image, "Olasilik", (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # İstatistik bilgileri
    bone_area = area_results['bone_area_mm2']
    bone_pixels = area_results['bone_pixels']
    bone_percentage = area_results['bone_percentage']
    pixel_area = area_results['pixel_area_mm2']
    total_area = area_results['total_area_mm2']
    
    # İstatistik paneli
    stats_height = 120
    stats_image = np.ones((stats_height, combined_image.shape[1], 3), dtype=np.uint8) * 240
    
    stats_lines = [
        f"Kemik Alani: {bone_area:.1f} mm²",
        f"Kemik Piksel: {bone_pixels:,} ({bone_percentage:.1f}%)",
        f"Piksel Alani: {pixel_area:.6f} mm²",
        f"Toplam Alan: {total_area:.0f} mm²"
    ]
    
    y_offset = 25
    for line in stats_lines:
        cv2.putText(stats_image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += 25
    
    # Tüm görüntüleri birleştir
    final_image = np.vstack([title_image, combined_image, stats_image])
    
    # Kaydet
    cv2.imwrite('simple_test_result.png', cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print("Gorsel kaydedildi: simple_test_result.png")
    
    # Görüntüyü göster
    cv2.imshow('Kemik Segmentasyon Testi', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_model(test_image_path=None, model_path=None):
    """Basit model testi"""
    
    print("Model testi basliyor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model yolu - komut satırı argümanı veya varsayılan
    if model_path is None:
        model_path = 'best_bone_model.pth'
    
    print(f"Model yolu: {model_path}")
    
    # Model oluştur
    print("Model olusturuluyor...")
    model = deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)  # 2 classes
    # aux_classifier'ı oluştur
    model.aux_classifier = nn.Sequential(
        nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(256, 2, kernel_size=1)
    )
    
    # Checkpoint yükle
    if os.path.exists(model_path):
        print("Model yukleniyor...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("OK - Model yuklendi!")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'val_iou' in checkpoint:
            print(f"  Val IoU: {checkpoint.get('val_iou'):.4f}")
    else:
        print("WARNING - Model dosyasi bulunamadi!")
        return
    
    # Custom forward method to avoid aux issues
    def forward_without_aux(self, x):
        input_shape = x.shape[-2:]
        # Get features
        features = self.backbone(x)
        # Use only the main classifier
        result = self.classifier(features['out'])
        result = torch.nn.functional.interpolate(result, size=input_shape, mode='bilinear', align_corners=False)
        return {'out': result}
    
    # Replace the forward method
    model.forward = forward_without_aux.__get__(model, type(model))
    
    model.to(device)
    model.eval()
    
    # Test görüntüsü - komut satırı argümanı veya varsayılan
    if test_image_path is None:
        test_image = 'merged/sag412.jpg'
    else:
        test_image = test_image_path
    
    if not os.path.exists(test_image):
        print(f"ERROR - Test goruntusu bulunamadi: {test_image}")
        return
    
    print(f"Test goruntusu: {test_image}")
    
    # Görüntüyü hazırla
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    print("Tahmin yapiliyor...")
    
    # Tahmin yap
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        model_output = model(image_tensor)
        print(f"Model output keys: {list(model_output.keys())}")
        
        if 'out' in model_output:
            output = model_output['out']
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            prob_map = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
            
            print("Tahmin basarili!")
            print(f"Pred mask shape: {pred_mask.shape}")
            print(f"Prob map shape: {prob_map.shape}")
            print(f"Unique values in pred_mask: {np.unique(pred_mask)}")
            print(f"Prob map range: {prob_map.min():.3f} - {prob_map.max():.3f}")
            
            # Alan hesaplaması
            area_results = calculate_area_mm2(pred_mask, prob_map, image_size=(512, 512), real_size_mm=(168.96, 168.96))
            
            # Basit görselleştirme (PIL/OpenCV kullanarak)
            create_simple_visualization(image, pred_mask, prob_map, area_results, test_image)
            
            # Detaylı görselleştirme - alan bilgisi ile
            create_detailed_visualization(image, pred_mask, prob_map, area_results, test_image)
            
            print("Gorsel kaydedildi: simple_test_result.png")
            
        else:
            print("ERROR - 'out' key bulunamadi!")
    
    print("Test tamamlandi!")

if __name__ == "__main__":
    import sys
    
    try:
        # Komut satırı argümanlarını kontrol et
        if len(sys.argv) > 2:
            test_image_path = sys.argv[1]
            model_path = sys.argv[2]
            print(f"Komut satiri argumanlari:")
            print(f"  Goruntu: {test_image_path}")
            print(f"  Model: {model_path}")
            test_model(test_image_path, model_path)
        elif len(sys.argv) > 1:
            test_image_path = sys.argv[1]
            print(f"Komut satiri argumani (goruntu): {test_image_path}")
            test_model(test_image_path)
        else:
            test_model()
    except Exception as e:
        print(f"HATA: {e}")
        print("Detayli hata bilgisi:")
        import traceback
        traceback.print_exc()
        input("Devam etmek icin Enter'a basin...")
        sys.exit(1)
