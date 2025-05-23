import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import os
import json
import logging
import concurrent.futures
from typing import Dict, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcript_summary.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("transcript_summarizer")

INDEX_URL = "https://lawwu.github.io/transcripts/index_langchain_interrupt_2025_official.html"
TRANSCRIPT_BASE = "https://lawwu.github.io/transcripts/"

# Map of YouTube IDs to their official titles
video_id_to_title = {
    "DrygcOI-kG8": "Interrupt 2025 Keynote",
    "fegwPmaAPQk": "Alice 2: Building and Scaling an AI Agent",
    "H-1QaLPnGsg": "Building Reliable Agents: Lessons in Building an IDE",
    "paaOevEFNlo": "Building Reliable Agents: Evaluation Challenges",
    "KfXq9s96tPU": "Multi-Agent Frontiers: Making Devin",
    "Bugs0dVcNI8": "From Pilot to Platform: Agentic Developer Products",
    "h_oUYqkRybM": "Building Replit Agent v2",
    "yMalr0jiOAc": "Multi-Agent Frontiers: Building Ask D.A.V.I.D.",
    "1PRcceHpJjM": "Breakthrough Agents: Building Reliable Agentic Systems",
    "NmblVxyBhi8": "From Pilot to Platform: Agents at Scale with LangGraph",
    "pKk-LfhujwI": "Breakthrough Agents: Learnings from Building AI Research Agents",
    "gPhyPRtIMn0": "Multi-Agent Frontiers: Transforming Customer Experience with Cisco",
    "kuXtW03cZEA": "Building Reliable Agents: Raising the Bar",
    "uNBIaANTJJw": "Unlocking Agent Creation: Agentic Architecture Lessons",
    "P8ewpJrZVwo": "How Monday.com Built Their Digital Workforce",
    "_XWJdCZM8Ag": "From LLMs to Agents: The Next Leap",
    "4pYzYmSdSH4": "State of Agents with Andrew Ng",
    "DsjkO2vB618": "Building Reliable Agents: Agent Evaluations",
}

# Get the list of video IDs in the preferred order
video_ids = list(video_id_to_title.keys())

# 1. Scrape all transcript URLs, YouTube IDs, and talk names from the index page
def get_transcript_info(index_url) -> Dict[str, Dict[str, str]]:
    logger.info(f"Fetching transcript information from {index_url}")
    resp = requests.get(index_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Initialize our dictionary to store YouTube ID -> {url, title} mappings
    talk_info = {}
    
    # Find all rows in the table to get titles
    rows = soup.find_all('tr')
    logger.info(f"Found {len(rows)-1} potential transcript rows in the index")
    
    # Map of YouTube IDs to titles
    id_to_title = {}
    
    # Skip the header row
    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) >= 5:  # Make sure we have enough cells
            title = cells[1].text.strip()
            # Get the transcript URL from the fifth cell (Transcript Only)
            transcript_link = cells[4].find('a', href=True)
            
            if transcript_link and transcript_link['href'].startswith("transcript_"):
                # Extract YouTube ID from the transcript URL
                youtube_id = transcript_link['href'].replace("transcript_", "").replace(".html", "")
                id_to_title[youtube_id] = title
    
    # If we couldn't extract titles from the index, use our predefined titles
    if not id_to_title and len(video_ids) > 0:
        logger.warning("Could not extract titles from index, using predefined titles")
        id_to_title = video_id_to_title.copy()
    
    # Create the talk_info dictionary with standardized URLs
    for youtube_id, title in id_to_title.items():
        transcript_url = f"{TRANSCRIPT_BASE}transcript_{youtube_id}.html"
        talk_info[youtube_id] = {
            "title": title,
            "url": transcript_url
        }
        logger.debug(f"Added talk: {title} (ID: {youtube_id})")
    
    logger.info(f"Successfully extracted information for {len(talk_info)} talks")
    return talk_info

# 2. Download and extract main transcript text from a transcript page
def get_transcript_text(transcript_url):
    logger.info(f"Downloading transcript from {transcript_url}")
    resp = requests.get(transcript_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Try to get the main transcript div or fallback to body text
    transcript_div = soup.find("div", {"class": "transcript"})
    if transcript_div:
        text = transcript_div.get_text(separator="\n", strip=True)
        logger.info(f"Found transcript div with {len(text)} characters")
    else:
        text = soup.get_text(separator="\n", strip=True)
        logger.info(f"No transcript div found, using full page text ({len(text)} characters)")
    
    return text

# 3. Summarize using Gemini-2.0-flash via LangChain
def summarize_text(text, llm):
    logger.info(f"Summarizing transcript with {len(text)} characters")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert meeting summarizer."),
        ("human", "Summarize the following transcript with the key points and main takeaways.\n\n{text}")
    ])
    chain = prompt | llm
    
    logger.debug("Invoking LLM for summarization")
    response = chain.invoke({"text": text})
    summary = response.content if hasattr(response, 'content') else str(response)
    logger.info(f"Generated summary with {len(summary)} characters")
    
    return summary

# Function to process a single transcript - for parallel execution
def process_single_transcript(youtube_id, info, llm):
    try:
        logger.info(f"Processing talk: '{info['title']}' (ID: {youtube_id})")
        text = get_transcript_text(info['url'])
        summary = summarize_text(text, llm)
        logger.info(f"Successfully summarized talk: '{info['title']}'")
        return youtube_id, summary, None
    except Exception as e:
        error_msg = f"Error processing {info['url']}: {str(e)}"
        logger.error(error_msg)
        return youtube_id, "Error: Failed to generate summary", str(e)

def main():
    logger.info("Starting transcript summarization process")
    
    logger.info("Initializing Vertex AI LLM")
    llm = ChatVertexAI(model_name="gemini-2.0-flash-001")  # Set project/location if needed

    # Get talk information (YouTube IDs, titles, and URLs)
    talk_info = get_transcript_info(INDEX_URL)
    logger.info(f"Found {len(talk_info)} transcripts")
    
    # Process transcripts in parallel using ThreadPoolExecutor
    # Using threads instead of processes because this is an I/O-bound task
    logger.info("Starting parallel transcript summarization")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all jobs
        future_to_id = {}
        for youtube_id, info in talk_info.items():
            future = executor.submit(process_single_transcript, youtube_id, info, llm)
            future_to_id[future] = youtube_id
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_id), 
                          total=len(future_to_id), 
                          desc="Summarizing transcripts"):
            youtube_id = future_to_id[future]
            try:
                id, summary, error = future.result()
                talk_info[youtube_id]["summary"] = summary
                if error:
                    talk_info[youtube_id]["error"] = error
            except Exception as e:
                logger.error(f"Exception processing result for {youtube_id}: {str(e)}")
                talk_info[youtube_id]["summary"] = "Error: Exception during processing"
                talk_info[youtube_id]["error"] = str(e)
    
    logger.info("Completed parallel summarization")
    
    # Output summaries in multiple formats
    logger.info("Writing results to files")
    
    # JSON format for programmatic access
    with open("summaries.json", "w") as f:
        json.dump(talk_info, f, indent=2)
        logger.info("Saved summaries to summaries.json")
    
    # Text format for human readability
    with open("summaries.txt", "w") as f:
        for youtube_id, info in talk_info.items():
            f.write(f"YouTube ID: {youtube_id}\n")
            f.write(f"Title: {info['title']}\n")
            f.write(f"URL: {info['url']}\n")
            f.write(f"Summary: {info.get('summary', 'No summary available')}\n")
            f.write(f"{'-'*80}\n")
        logger.info("Saved summaries to summaries.txt")
    
    # Markdown format for blog publication
    logger.info("Generating markdown file for blog publication")
    with open("langchain_interrupt_2025_summaries.md", "w") as f:
        # Write markdown header
        f.write("# LangChain Interrupt 2025 - AI-Generated Talk Summaries\n\n")
        f.write("*This page contains AI-generated summaries of the LangChain Interrupt 2025 conference talks.*\n\n")
        
        # Process videos in a specific order (can use video_ids list if order matters)
        for youtube_id in video_ids:
            if youtube_id not in talk_info:
                logger.warning(f"No information found for video ID: {youtube_id}")
                continue
                
            info = talk_info[youtube_id]
            
            # Use the official title from our mapping instead of what was scraped
            title = video_id_to_title.get(youtube_id, info['title'])
            
            # Talk title as heading
            f.write(f"## {title}\n\n")
            
            # Video embed using shortcode
            youtube_link = f"https://www.youtube.com/watch?v={youtube_id}"
            f.write(f"{{{{< video {youtube_link} >}}}}\n\n")
            
            # Link to transcript
            f.write(f"Transcript: <{info['url']}>\n\n")
            
            # AI summary
            f.write("AI Summary:\n\n")
            f.write(f"{info.get('summary', 'No summary available')}\n\n")
            
            # Separator
            f.write("---\n\n")
        
        logger.info("Saved markdown file to langchain_interrupt_2025_summaries.md")
    
    logger.info("Transcript summarization process completed successfully")

if __name__ == "__main__":
    try:
        logger.info("Script execution started")
        main()
        logger.info("Script execution completed successfully")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        raise
