# Module 3: Build a Simple Eval in the Braintrust UI

## Assets

- **`customer_complaints.csv`** — 16 customer support messages to upload to Braintrust as a dataset.
- **`prompt_a_polite.txt`** — System prompt for the polite persona.
- **`prompt_b_concise.txt`** — System prompt for the concise persona.
- **`scorer.txt`** — LLM-as-a-judge scorer prompt.

## Prerequisites

### Create a Braintrust account

1. Go to [braintrust.dev](https://www.braintrust.dev) and sign up. The free tier covers everything in this course.

### Add your OpenAI API key to Braintrust

The playground needs an OpenAI key to make LLM calls. In the Braintrust UI:

1. Go to **Settings → Secrets**
2. Add your OpenAI API key

This lets the playground and online scoring call OpenAI on your behalf.

## Setup in the Braintrust UI

### 1. Create the project

1. Click **New Project**
2. Name it **"Customer Support Chatbot"**

This project is used for the rest of the course.

### 2. Upload the dataset

1. Inside the project, click **Datasets → New Dataset**
2. Name it **"Customer Support Messages"**
3. Upload `customer_complaints.csv` from this directory

### 3. Set up the playground

1. Click **Playground** in the left sidebar
2. In the **Project** field, select **"Customer Support Chatbot"**
3. Under **System**, paste the contents of `prompt_a_polite.txt`
4. Under **User**, enter `{{input}}`
5. Click **Add dataset** and select **"Customer Support Messages"**

### 4. Add a scorer

1. Click **Add scorer** and select **LLM as a judge**
2. Paste the contents of `scorer.txt` into the scorer prompt field
3. Under **Output type**, select **Score**
4. Under **Choice scores**, add the following:
   - Choice: `A` → Score: `1`
   - Choice: `B` → Score: `0.5`
   - Choice: `C` → Score: `0`
5. Turn on **Use chain of thought (CoT)**

### 5. Save experiments

1. Click **Run** to run Prompt A against all inputs in the dataset
2. Click **Save as experiment** and name it **`module_3_polite_persona`**
3. Swap the system prompt for the contents of `prompt_b_concise.txt`
4. Click **Run** again
5. Click **Save as experiment** and name it **`module_3_concise_persona`**

### 6. Compare experiments

1. Open the **Experiments** tab
2. Select both `module_3_polite_persona` and `module_3_concise_persona` and click "Compare"
3. Look at aggregate score differences and per-input diffs
