import os
import glob

os.makedirs("texts/processed", exist_ok=True)

for filepath in glob.glob("texts/raw/*.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
        words = f.read().split()
    
    name = os.path.splitext(os.path.basename(filepath))[0]
    chunk_size = 5000
    
    if len(words) <= chunk_size:
        # Copy small files as-is
        with open(f"texts/processed/{name}.txt", 'w', encoding='utf-8') as f:
            f.write(' '.join(words))
        print(f"{name}.txt: {len(words):,} words (copied)")
    else:
        # Split large files
        for i, start in enumerate(range(0, len(words), chunk_size)):
            chunk = words[start:start + chunk_size]
            with open(f"texts/processed/{name}_{i}.txt", 'w', encoding='utf-8') as f:
                f.write(' '.join(chunk))
            print(f"{name}_{i}.txt: {len(chunk):,} words")
        print(f"Split {name}.txt ({len(words):,} words) into {(len(words)-1)//chunk_size + 1} chunks")