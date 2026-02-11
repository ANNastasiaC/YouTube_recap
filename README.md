# YouTube_recap

Analyze your exported YouTube watch history and generate your own custom recap.

This script turns your Google Takeout watch history into a CSV with:

video_id | title | watch_time | category | description

From there, you can explore trends over time, keyword shifts, category changes, or even semantic patterns.


##How to Use

1, Download your YouTube watch history from https://takeout.google.com

2, Set in the script:

HTML_FILE = "path/to/watch-history.html"

API_KEY = "your_youtube_api_key"

3, Run the script.

##Included Examples

Keyword frequency analysis

<img width="1015" height="571" alt="Screenshot 2026-02-11 at 16 57 49" src="https://github.com/user-attachments/assets/eabfcad9-7998-484b-b980-b5ec46c83e61" />

Category distribution over time

<img width="904" height="470" alt="Screenshot 2026-02-11 at 16 58 04" src="https://github.com/user-attachments/assets/c9ecc5b1-e72e-4159-b18f-9cee2ad9ff18" />

Diversity (entropy) over time

<img width="725" height="405" alt="Screenshot 2026-02-11 at 16 58 13" src="https://github.com/user-attachments/assets/23134721-e965-4a0b-a59f-0584a1af5292" />

Semantic embedding map of descriptions

<img width="564" height="577" alt="Screenshot 2026-02-11 at 16 58 21" src="https://github.com/user-attachments/assets/37835e3a-f51f-462b-9622-fd2368dbf4eb" />

##The rest is up to you.
