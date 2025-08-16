# mini-gandalf

A simple but fun LLM red teaming little game, inspired by [gandalf.lakera.ai](https://gandalf.lakera.ai/).

## Setup

### Prerequisites

-   Docker

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/FumingPower3925/mini-gandalf.git](https://github.com/FumingPower3925/mini-gandalf.git)
    cd mini-gandalf
    ```

2.  **Set up the environment:**

    Create a `.env` file from the example and add your OpenAI API key:

    ```bash
    cp .env.example .env
    ```

    Edit the `.env` file and add your key.

3.  **Train the Models (Important!):**

    Before running the main application, you need to train the classifier models for levels 3 and 4. Run the following commands:

    ```bash
    docker compose run --rm train-input-classifier
    ```
    These commands will download the datasets and save the fine-tuned models to a new `models/` directory.

4.  **Build and run the main application:**

    ```bash
    docker compose up --build web
    ```

The application will be available at `http://localhost:7860`.