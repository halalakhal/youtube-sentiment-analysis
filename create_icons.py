from PIL import Image, ImageDraw, ImageFont

def create_icon(size):
    # Créer une image avec fond bleu
    img = Image.new('RGB', (size, size), color='#4285f4')
    draw = ImageDraw.Draw(img)
    
    # Dessiner un cercle blanc
    margin = size // 4
    draw.ellipse([margin, margin, size-margin, size-margin], fill='white')
    
    # Sauvegarder
    img.save(f'chrome-extension/icons/icon{size}.png')
    print(f'✓ Icône {size}x{size} créée')

# Créer les 3 tailles
for size in [16, 48, 128]:
    create_icon(size)

print('\n✅ Toutes les icônes sont créées!')
