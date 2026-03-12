1. Activate VE
source venv/bin/activate

2. Install dependencies 
pip install --upgrade pip
pip install -r requirements.txt

3. Login to Huggingface
huggingface-cli login

4. Start Training 
python main.py

5. After training run 
python inference.py
