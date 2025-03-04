# master-AI-LLM-Agents-Courses

## Setup instructions for Linux

1. **Install Anaconda:**
  - Download the Linux installer from https://www.anaconda.com/download.
  - Open a terminal and navigate to the folder containing the downloaded .sh file.
  - Run the installer: bash Anaconda3*.sh and follow the prompts. Note: This requires about 5+ GB of disk space.
  
2. **Set up the environment:**
  - Open a terminal and navigate to the "project root directory" using: cd ~/Projects/llm_engineering (adjust the path as necessary).
  - Run ls to confirm the presence of subdirectories for each week of the course.
  - Create the environment: conda env create -f environment.yml
     This may take several minutes (even up to an hour for new Anaconda users). If it takes longer or errors occur, proceed to Part 2B.
  - Activate the environment: conda activate llms.
     You should see (llms) in your prompt, indicating successful activation.
