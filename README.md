\# Strategic Drone Base Optimization Simulation  

\*\*Equitable, Efficient Aerial-Asset Planning for Disaster Response\*\*

This is a simulation project based on the research paper:  

\*\*“Strategic Planning of Aerial Assets for Disaster Response: Enabling Efficient and Equitable Access to Drone-Based Search Resources”\*\*  

by Chin, Saravanan, Balakrishnan, and Andreeva-Mori (MIT & JAXA, ATM 2023).

It demonstrates how to strategically locate drone bases to ensure \*\*maximum coverage\*\*, \*\*minimum response time\*\*, and \*\*fair access\*\* to drone resources across a disaster-prone region.

\---

\## 🚁 Key Features

\- \*\*Grid-Based Mapping\*\*: The simulation divides the target region into grid cells and assigns a \*search need probability\* to each based on elevation, population density, flood zones, and historical data.

\- \*\*Optimization Engine\*\*: Uses a Mixed Integer Programming (MIP) model to select drone base locations that:

  - Maximize coverage,

  - Minimize Distance to Nearest Base (DNB),

  - Balance both objectives using a \*\*γ (gamma)\*\* parameter.

\- \*\*Equity Metrics\*\*: Calculates \*\*Gini Coefficient\*\* to quantify inequality in drone access across the region.

\- \*\*Base Failure Modeling (r)\*\*: Incorporates probability of base failure during a disaster.

\- \*\*Planned Relocation (PPR)\*\*: Optimizes for relocation in case some bases become inoperable.

\- \*\*Interactive Interface\*\*: Sliders to modify number of bases, drone range, gamma, uncertainty \`r\`, etc. Coverage, DNB, and equity maps update live.

\---

\## 📦 Installation

\`\`\`bash

git clone https\://github.com/your-org/drone-base-optimization.git

cd drone-base-optimization

python -m venv venv

source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

🚀 Running the Simulation

GUI Mode

bash

Copy

Edit

streamlit run app.py

CLI Mode

bash

Copy

Edit

python cli.py --num-bases 5 --range 30 --gamma 0.5 --uncertainty 0.1

⚙️ Parameters Guide

Parameter Description Typical Range

\--num-bases Number of drone bases to deploy 1 - 25

\--range Drone range in km (1-way) 5 - 50

\--gamma Trade-off between coverage and response time (0–1) 0.0 - 1.0

\--uncertainty Probability a base is inoperable due to disaster 0.0 - 0.3

\--seed Seed for reproducibility Any integer

🧠 How the Model Works

Grid Setup: Input rasters (population, terrain, risk) → composite probability heatmap.

Optimization Objective:

𝑍

\=

−

Coverage

\+

𝛾

∑

𝑖

∈

covered

𝑝

𝑖

⋅

DNB

𝑖

Z=−Coverage+γ 

i∈covered

∑

​

 p 

i

​

 ⋅DNB 

i

​

 

Equity Check: Gini coefficient of DNB values across grid cells.

Base Uncertainty: Model base failure via Bernoulli trials (r).

PPR Stage: Rerun optimization after disaster effects to reposition selected bases.

📂 Directory Structure

bash

Copy

Edit

.

├── app.py             # Streamlit frontend

├── cli.py             # Command-line interface

├── model/

│   ├── optimizer.py   # Optimization formulation

│   ├── grid.py        # Grid setup and probability assignment

│   ├── equity.py      # Gini calculation

│   └── relocation.py  # Planned relocation logic

├── data/              # Sample raster inputs

├── docs/              # Model explanation and screenshots

└── tests/             # Unit tests

📖 Citation

If you use this simulation in coursework or academic projects, cite:

Chin, C., Saravanan, A., Balakrishnan, H., Andreeva-Mori, A.

"Strategic Planning of Aerial Assets for Disaster Response: Enabling Efficient and Equitable Access to Drone-Based Search Resources."

ATM R\&D Seminar, 2023 (Tokyo, Japan).

bibtex

Copy

Edit

@inproceedings{chin2023strategic,

  title={Strategic Planning of Aerial Assets for Disaster Response: Enabling Efficient and Equitable Access to Drone-Based Search Resources},

  author={Chin, Christopher and Saravanan, Akila and Balakrishnan, Hamsa and Andreeva-Mori, Adriana},

  booktitle={ATM R\\\&D Seminar},

  year={2023}

}

📜 License

This simulation is distributed under the MIT License.

🙌 Acknowledgements

MIT Department of Aeronautics and Astronautics

JAXA (Japan Aerospace Exploration Agency)

Open-source tools: PuLP, GeoPandas, Streamlit, OR-Tools, Folium
