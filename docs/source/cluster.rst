Running in Jupyter Notebook on a HPC platform
=============================================

.. note::
   This example is based on CSD3 for people associated with the University of Cambridge but the workflow can be applied to other HPC clusters.

This guide provides step-by-step instructions on how to set up a virtual environment for Jupyter Notebook on an HPC system, run Jupyter Notebook on both login and compute nodes, install the ``detectree2`` package, and run it on a GPU compute node.

.. contents::


Setting Up a Virtual Environment for Jupyter Notebook
-----------------------------------------------------

First, create a virtual environment to run Jupyter Notebook. This step only needs to be done once.

.. code-block:: console

   # Load the Python module
   module load python/3.8

   # Create a virtual environment with system site packages
   virtualenv --system-site-packages ~/jupyter-env

   # Activate the virtual environment
   source ~/jupyter-env/bin/activate

   # Install Jupyter Notebook
   pip install jupyter




Running Jupyter Notebook on the Login Node (CPU Only)
-----------------------------------------------------

Follow these steps to run Jupyter Notebook on the login node using CPU resources only.

**Steps:**
1. **Activate the Virtual Environment:**

   .. code-block:: console

      source ~/jupyter-env/bin/activate

2. **Start Jupyter Notebook:**

   .. code-block:: console

      jupyter notebook --no-browser --ip=127.0.0.1 --port=8081

   - This command will start a Jupyter Notebook server and output a URL that looks like ``http://127.0.0.1:8081/?token=...``. Save this URL for later use.
   - If the ``127.0.0.1`` URL doesn't work, try using the one that starts with ``http://localhost``.

3. **Set Up SSH Tunneling from Your Local Computer:**

   Open a terminal on your local computer and run:

   .. code-block:: console

      ssh -L 8081:127.0.0.1:8081 -N <CRSid>@<login_hostname>.hpc.cam.ac.uk

   - Replace ``<CRSid>`` with your user ID and ``<login_hostname>`` with the hostname of the login node (e.g., ``login-p-1``).
   - Example:

     .. code-block:: console

        ssh -L 8081:127.0.0.1:8081 -N ab123@login-p-1.hpc.cam.ac.uk

   - Keep this terminal open as long as you want the session to last.

4. **Access Jupyter Notebook:**

   - Paste the URL you saved earlier into your web browser to access the Jupyter Notebook interface.



Running Jupyter Notebook on a Compute Node (CPU or GPU)
-------------------------------------------------------

To run Jupyter Notebook on a compute node, follow these steps.

**Steps:**
1. **Note the Login Node Hostname:**

   .. code-block:: console

      hostname

   - Run this command on the login node and note the hostname (e.g., ``login-p-1``).

2. **Request an Interactive Session:**

   - **For GPU Compute Node:**

     .. code-block:: console

        sintr -A COOMES-SL3-GPU -p ampere -N1 -n1 --gres=gpu:1 -t 1:0:0 --qos=INTR

   - **For CPU Compute Node:**

     .. code-block:: console

        sintr -A COOMES-SL3-CPU -p icelake -N1 -n38 -t 1:0:0 --qos=INTR

     - Note that ``COOMES-SL3-*`` should be replaced with your or your lab's balance account.

3. **Note the Compute Node Hostname:**

   .. code-block:: console

      hostname

   - Run this command on the compute node and note the hostname (e.g., ``gpu-q-3``).

4. **Activate the Virtual Environment:**

   .. code-block:: console

      source ~/jupyter-env/bin/activate

5. **Load Necessary Modules (if required):**

   - For example, to load CUDA:

     .. code-block:: console

        module load cuda

6. **Start Jupyter Notebook:**

   .. code-block:: console

      jupyter notebook --no-browser --ip=* --port=8081

   - This will output a URL starting with ``http://127.0.0.1:8081/?token=...``. Save this URL for later.

7. **Set Up SSH Tunneling from Your Local Computer:**

   Open a terminal on your local computer and run:

   .. code-block:: console

      ssh -L 8081:<compute_hostname>:8081 -N <CRSid>@<login_hostname>.hpc.cam.ac.uk

   - Replace ``<compute_hostname>`` with the hostname from step 3 and ``<login_hostname>`` from step 1.
   - Example:

     .. code-block:: console

        ssh -L 8081:gpu-q-3:8081 -N ab123@login-p-1.hpc.cam.ac.uk

   - Keep this terminal open as long as you want the session to last.

8. **Access Jupyter Notebook:**

   - Paste the URL you saved earlier into your web browser to access the Jupyter Notebook interface.

.. note::
   If you are connected to the cluster via VSCode, it will try to forward the port automatically, which does not work. In that case you might need to shuffle ports around and connect to the new port, like: ``ssh -L 8082:gpu-q-3:8081 ...`` and then enter the link in your browser by replacing 8081 with 8082.


Installing Detectree2
---------------------

Follow these steps to install the ``detectree2`` package on a GPU compute node.

**Steps:**
1. **Request a GPU Interactive Session:**

   .. code-block:: console

      sintr -A COOMES-SL3-GPU -p ampere -N1 -n1 --gres=gpu:1 -t 0:30:0 --qos=INTR

2. **Load Required Modules:**

   .. code-block:: console

      module load gcc
      module load cuda/12.1
      module load cudnn/8.9_cuda-12.1
      module load gdal/3.7.0-icl

3. **Activate the Virtual Environment:**

   .. code-block:: console

      source ~/jupyter-env/bin/activate

4. **Install PyTorch with CUDA Support:**

   .. code-block:: console

      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

5. **Install Additional Python Packages:**

   .. code-block:: console

      pip install opencv-python
      pip install GDAL==3.7.0

6. **Set CUDA Environment Variables:**

   .. code-block:: console

      export CUDA_HOME=/usr/local/software/cuda/12.1
      export PATH=/usr/local/software/cuda/12.1/bin:/usr/local/software/cuda/12.1/cuda-samples-12.1/bin/x86_64/linux/release:/usr/local/software/cuda/12.1/nvvm/bin:/usr/local/software/cuda/12.1/libnvvp:$PATH
      export CPATH=/usr/local/software/cuda/12.1/include:$CPATH
      export FPATH=/usr/local/software/cuda/12.1/include:$FPATH
      export LIBRARY_PATH=/usr/local/software/cuda/12.1/lib64:/usr/local/software/cuda/12.1/lib:/usr/local/software/cuda/12.1/nvvm/lib64:$LIBRARY_PATH
      export LD_LIBRARY_PATH=/usr/local/software/cuda/12.1/lib64:/usr/local/software/cuda/12.1/lib:/usr/local/software/cuda/12.1/nvvm/lib64:$LD_LIBRARY_PATH

7. **Install Detectree2:**

   .. code-block:: console

      pip install git+https://github.com/PatBall1/detectree2.git



Running Detectree2 on a GPU Compute Node
----------------------------------------

When running any ``detectree2`` tasks on a GPU compute node, ensure the following modules are loaded and the Python environment is active.

**Steps:**
1. **Request a GPU Interactive Session:**

   .. code-block:: console

      sintr -A COOMES-SL3-GPU -p ampere -N1 -n1 --gres=gpu:1 -t 1:0:0 --qos=INTR

2. **Load Required Modules:**

   .. code-block:: console

      module load cuda/12.1
      module load cudnn/8.9_cuda-12.1
      module load gdal/3.7.0-icl

3. **Activate the Virtual Environment:**

   .. code-block:: console

      source ~/jupyter-env/bin/activate

4. **Run Your Detectree2 Tasks:**

   - You can now run your Python scripts or Jupyter Notebooks that utilize ``detectree2``.


.. note::
   Always ensure that you're working within the allocated time for interactive sessions and that you comply with the HPC usage policies.
