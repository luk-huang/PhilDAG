#!/usr/bin/env python3
"""Download philosophy texts from Project Gutenberg"""
import os
import requests
import time
from pathlib import Path

def download_text(url):
    """Download text from URL"""
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        r.encoding = 'utf-8'
        return r.text if len(r.text) > 100 else None
    except:
        return None

def clean_text(text, start, end, start_offset=0, end_offset=0):
    """Clean text using markers"""
    if not text:
        return None
    
    if start and start in text:
        text = text.split(start)[1][start_offset:]
    
    if end and end in text:
        text = text.split(end)[0]
        if end_offset < 0:
            text = text[:end_offset]
    
    return text.strip()

def save_text(text, filename, output_dir="texts"):
    """Save text to file"""
    if not text:
        return False
    
    Path(output_dir).mkdir(exist_ok=True)
    file_path = Path(output_dir) / f"{filename}.txt"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return True

# All texts to download
TEXTS = [
    # Rationalism
    {
        'name': 'spinoza_ethics',
        'url': 'http://www.gutenberg.org/cache/epub/3800/pg3800.txt',
        'start': 'ranslated from the Latin by R.',
        'end': 'End of the Ethics',
        'start_offset': 71
    },
    {
        'name': 'spinoza_improve_understanding',
        'url': 'http://www.gutenberg.org/cache/epub/1016/pg1016.txt',
        'start': 'Farewell.*',
        'end': 'End of',
        'start_offset': 20
    },
    {
        'name': 'leibniz_theodicy',
        'url': 'http://www.gutenberg.org/cache/epub/17147/pg17147.txt',
        'start': 'appeared in 1710 as the',
        'end': 'SUMMARY OF THE CON',
        'start_offset': 202,
        'end_offset': -140
    },
    {
        'name': 'descartes_discourse_method',
        'url': 'http://www.gutenberg.org/cache/epub/59/pg59.txt',
        'start': 'PREFATORY NOTE',
        'end': 'End of the Pr',
        'start_offset': 18
    },
    # Empiricism
    {
        'name': 'locke_understanding_vol1',
        'url': 'http://www.gutenberg.org/cache/epub/10615/pg10615.txt',
        'start': '2 Dorset Court, 24th of May, 1689',
        'end': 'End of the Pro',
        'start_offset': 50,
        'end_offset': -30
    },
    {
        'name': 'locke_understanding_vol2',
        'url': 'http://www.gutenberg.org/cache/epub/10616/pg10616.txt',
        'start': '1. Man fitted to form articulated Sounds.',
        'end': 'End of the Pro',
        'start_offset': 4,
        'end_offset': -25
    },
    {
        'name': 'locke_treatise_government',
        'url': 'http://www.gutenberg.org/cache/epub/7370/pg7370.txt',
        'start': 'now lodged in Christ College, Cambridge.',
        'end': 'FINIS.',
        'start_offset': 21
    },
    {
        'name': 'hume_treatise',
        'url': 'http://www.gutenberg.org/cache/epub/4705/pg4705.txt',
        'start': 'ADVERTISEMENT',
        'end': 'End of Pro',
        'start_offset': 9,
        'end_offset': -14
    },
    {
        'name': 'hume_natural_religion',
        'url': 'http://www.gutenberg.org/cache/epub/4583/pg4583.txt',
        'start': 'PAMPHILUS TO HERMIPPUS',
        'end': 'End of the Pro',
        'start_offset': 6,
        'end_offset': -22
    },
    {
        'name': 'berkeley_treatise',
        'url': 'http://www.gutenberg.org/cache/epub/4723/pg4723.txt',
        'start': 'are too apt to condemn an opinion before they rightly',
        'end': 'End of the Pr',
        'start_offset': 47,
        'end_offset': -22
    },
    {
        'name': 'berkeley_three_dialogues',
        'url': 'http://www.gutenberg.org/cache/epub/4724/pg4724.txt',
        'start': 'THE FIRST DIALOGUE',
        'end': 'End of the Pro',
        'start_offset': 17,
        'end_offset': -22
    },
    # Analytic
    {
        'name': 'russell_problems_philosophy',
        'url': 'http://www.gutenberg.org/cache/epub/5827/pg5827.txt',
        'start': 'n the following pages',
        'end': 'BIBLIOGRAPHICAL NOTE'
    },
    {
        'name': 'russell_analysis_mind',
        'url': 'http://www.gutenberg.org/cache/epub/2529/pg2529.txt',
        'start': 'H. D. Lewis',
        'end': 'End of Pro',
        'start_offset': 21
    },
    {
        'name': 'moore_philosophical_studies',
        'url': 'http://www.gutenberg.org/files/50141/50141-0.txt',
        'start': 'Aristotelian Society,_ 1919-20.',
        'end': 'E Wes',
        'start_offset': 23,
        'end_offset': -10
    },
    # Political Economy
    {
        'name': 'smith_wealth_nations',
        'url': 'http://www.gutenberg.org/files/3300/3300-0.txt',
        'start': 'INTRODUCTION AND PLAN OF THE WORK.',
        'end': 'End of the Project Gutenberg EBook of An Inquiry into the Nat'
    },
    {
        'name': 'ricardo_political_economy',
        'url': 'http://www.gutenberg.org/cache/epub/33310/pg33310.txt',
        'start': 'ON VALUE.',
        'end': 'FOOTNOTES:'
    }
]

def main():
    print("Downloading philosophy texts from Project Gutenberg...")
    
    downloaded = 0
    failed = []
    
    for book in TEXTS:
        print(f"  {book['name']}...", end=' ')
        
        text = download_text(book['url'])
        
        if text:
            cleaned = clean_text(
                text,
                book.get('start', ''),
                book.get('end', ''),
                book.get('start_offset', 0),
                book.get('end_offset', 0)
            )
            
            if save_text(cleaned, book['name']):
                print("✓")
                downloaded += 1
            else:
                print("✗")
                failed.append(book['name'])
        else:
            print("✗")
            failed.append(book['name'])
        
        time.sleep(0.5)  # Be nice to servers
    
    print(f"\nDownloaded: {downloaded}/{len(TEXTS)} texts to ./texts/")
    if failed:
        print(f"Failed: {', '.join(failed)}")

if __name__ == "__main__":
    main()