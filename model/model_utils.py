# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 14:16:59 2025

@author: hussain
"""

SUBSET_NAMES = {
    "lung":["lung_aca", "lung_n", "lung_scc"],
    "colon":["colon_aca", "colon_n"],
    "scabies":["HEALTHY", "SCAB"],
    "matek":["Basophil", "Erythroblast", "Eosinophil", "Smudge cell", "Atypical Lymphocyte", "Typical Lymphocyte", 
              "Metamyelocyte", "Monoblast", "Monocyte", "Myelocyte", "Myeloblast", "Band Neutrophil", 
              "Segmented Neutrophil", "Promyelocyte Bilobed", "Promyelocyte"]
    }

TEMPLATES_SMALL = [
    "a {}photo of a {}",
    "a {}rendering of a {}",
    "a {}cropped photo of the {}",
    "the {}photo of a {}",
    "a {}photo of a clean {}",
    "a {}photo of a dirty {}",
    "a dark {}photo of the {}",
    "a {}photo of my {}",
    "a {}photo of the cool {}",
    "a close-up {}photo of a {}",
    "a bright {}photo of the {}",
    "a cropped {}photo of a {}",
    "a {}photo of the {}",
    "a good {}photo of the {}",
    "a {}photo of one {}",
    "a close-up {}photo of the {}",
    "a {}rendition of the {}",
    "a {}photo of the clean {}",
    "a {}rendition of a {}",
    "a {}photo of a nice {}",
    "a good {}photo of a {}",
    "a {}photo of the nice {}",
    "a {}photo of the small {}",
    "a {}photo of the weird {}",
    "a {}photo of the large {}",
    "a {}photo of a cool {}",
    "a {}photo of a small {}",
]

PROMPTS_BY_CLASS = {
    "colon_aca": "Ultra-high-resolution photorealistic histopathology slide of human colon adenocarcinoma tissue, hematoxylin and eosin (H&E) stained, microscopic view at 40x magnification, showing irregular glandular architecture, malignant epithelial cells with hyperchromatic nuclei, prominent nucleoli, high nuclear-to-cytoplasmic ratio, loss of normal crypt structure, desmoplastic stroma, visible mitotic figures, pink eosinophilic cytoplasm, blue-purple nuclear staining, intricate microvascular structures, sharp focus with ultra-detailed cellular morphology, scientifically accurate pathology style, vibrant yet realistic H&E color balance, aspect ratio 1:1 or 4:3, resolution 1024x1024, varied tumor grades and gland formation patterns",
    "colon_n": "Ultra-high-resolution photorealistic histopathology slide of normal benign colonic mucosa, hematoxylin and eosin (H&E) stained, microscopic view at 40x magnification, showing well-organized crypt architecture with uniform tubular glands, evenly spaced goblet cells, nuclei small and basally located, low nuclear-to-cytoplasmic ratio, intact mucosal epithelium, normal lamina propria with sparse inflammatory cells, pink eosinophilic cytoplasm, blue-purple nuclear staining, sharp focus with ultra-detailed cellular morphology, scientifically accurate pathology style, vibrant yet realistic H&E color balance, aspect ratio 1:1 or 4:3, resolution 1024x1024",
}

# =============================================================================
# PROMPTS_BY_CLASS = {
#     "HEALTHY": "Ultra high-resolution microscopic image of human skin, detailed cellular structure, epidermis and dermis layers visible, realistic textures, soft lighting, biomedical photography style, intricate skin cells and capillaries, shallow depth of field, extreme close-up, 8k resolution, scientific illustration, photorealistic",
#     "SCAB": "Ultra high-resolution microscopic image of human skin infected with scabies, detailed visualization of epidermis and dermis layers, visible Sarcoptes scabiei mites burrowed in the stratum corneum, skin irritation, inflamed tissue, cellular damage, red patches, detailed mite anatomy including legs and body segments, biomedical microscopy style, scientific imaging, photorealistic textures, 8K resolution, extreme close-up, shallow depth of field, clinical dermatology reference",
#     }
# =============================================================================
