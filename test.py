from rapidocr_pdf import RapidOCRDocLoader

loader = RapidOCRDocLoader("/home/songhau/Documents/VulnBot/data/knowledge_base/kbfp/content/Definitive_Guide_to_Penetration_Testing.pdf")
docs = loader.load()
print(docs)