import re
import requests
import tiktoken

def _clean_wiki_content(text: str) -> str:
    """Strip Wikipedia/Jina noise while preserving actual content and tables."""
    # Remove Jina reader metadata/header (everything before first real heading)
    jina_header = re.search(r'^.*?(?=^#\s)', text, flags=re.DOTALL | re.MULTILINE)
    if jina_header:
        text = text[jina_header.end():]
    # Remove "See also", "References", "External links", "Notes", "Further reading"
    # and everything after them (these are always at the bottom)
    text = re.sub(
        r'\n#{1,3}\s*(See also|References|External links|Notes|Further reading|'
        r'Bibliography|Sources|Cited sources).*',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )
    # Remove image markdown ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove inline links but keep text: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()

def count_tokens(text: str, model="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

def main():
    urls = [
        "https://en.wikipedia.org/wiki/1928_Summer_Olympics",
        "https://en.wikipedia.org/wiki/Wikipedia:Featured_article_candidates/Giganotosaurus/archive1",
        "https://en.wikipedia.org/wiki/Mercedes_Sosa"
    ]

    for url in urls:
        print(f"Fetching: {url}")
        
        # Prepend jina reader URL
        jina_url = f"https://r.jina.ai/{url}"
        
        try:
            response = requests.get(jina_url)
            response.raise_for_status()
            original_text = response.text
            
            cleaned_text = _clean_wiki_content(original_text)
            
            orig_tokens = count_tokens(original_text)
            clean_tokens = count_tokens(cleaned_text)
            
            if orig_tokens > 0:
                reduction = ((orig_tokens - clean_tokens) / orig_tokens) * 100
            else:
                reduction = 0.0
                
            print(f"Original Tokens: {orig_tokens:,}")
            print(f"Cleaned Tokens:  {clean_tokens:,}")
            print(f"Reduction:       {reduction:.2f}%\n")
            
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}\n")

if __name__ == "__main__":
    main()