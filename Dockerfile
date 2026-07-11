FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install numba

COPY . .

RUN for i in "compute" "cholesky2" "go_fast"; do python scripts/run_benchmark.py -b $i -f numpy; done
RUN for i in "compute" "cholesky2" "go_fast"; do python scripts/run_benchmark.py -b $i -f numba; done
RUN python scripts/plot_lines.py
RUN python scripts/plot_results.py
