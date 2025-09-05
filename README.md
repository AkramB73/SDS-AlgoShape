# SDS-AlgoShape

Inital implementation of AlgoShape, uses a diffusion-based optimisation algorithm to iteratively evolve a collection of shapes to match a target image.

- **Dual Support:**
    - **Metal GPU Acceleration:** Optimized for Apple Silicon for highly efficient, low-power processing.
    - **Numba CPU Acceleration:** Falls back to a high-performance, multi-threaded Numba implementation on non-Mac systems (Windows/Linux) or if Metal is unavailable.

**Install the core requirements:**

```bash
pip install -r requirements.txt
```

**For Apple Silicon/Mac users (Optional, for GPU Acceleration):**
To enable the high-performance Metal backend, install the `metalcompute` library:

```bash
pip install metalcompute
```

## Usage

To run the program, configure the parameters inside the `example_run.py` script and then execute it.

1. **Open the file `example_run.py`**.
2. **Edit the Configuration section**:
    
    ```python
    # examples/example_run.py
    
    # --- Configuration ---
    INPUT_IMAGE_PATH = "/path/to/your/image.jpg" # IMPORTANT: Change this path
    OUTPUT_FILENAME  = "final_best_agent.png"
    
    ....
    ```
    
3. **Run the script** from your terminal:
    
    ```bash
    python example_run.py
    ```
    

The program will run with your settings and save the final image to the specified `OUTPUT_FILENAME`.


