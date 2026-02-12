You’re dealing with a **passive / server‑edition RTX 6000**, so the key requirement isn’t a specific fan model but achieving **enough *directed* airflow across the card’s heatsink** to keep a ~300–350 W (Ada) or up‑to‑600 W (Blackwell server) board within spec.

Below is how to think about **CFM**, **fan types**, and where to find **vendor guidance**.

---

### 1. Why CFM alone is misleading for this GPU

For a **passive server card**, what really matters is:

* **Airflow volume (CFM) through the card’s fin stack**
* **Static pressure** to push air through a dense heatsink and any front drive cages / filters
* **Airflow geometry**: the air must be **channeled along the length of the card**, not just stirred in the chassis

A 300–600 W passive GPU in a server chassis is typically cooled by **multiple high‑pressure fans** in a front‑to‑back tunnel. This is why RTX 6000 **Server Edition** is officially intended for pre‑qualified rack servers, not generic towers.

---

### 2. Approximate airflow requirement (engineering estimate)

NVIDIA/PNY generally don’t publish a simple “CFM required” number for these cards; instead they state “requires adequate chassis airflow” and validate specific server platforms (Supermicro, Lenovo, etc.).

To get in the right ballpark, you can use a standard electronic cooling estimate:

> **CFM ≈ (Power in watts × 3.16) / ΔT(°C)**

Where ΔT is allowed air temperature rise across the GPU.

Assume:

* **Board power**:  
  * RTX 6000 Ada passive derivative: ~300–320 W  
  * RTX PRO 6000 Blackwell Server Edition: up to **600 W**.
* Allowable **air temperature rise** across the card: 10–15 °C is typical for high‑performance server cooling.

For a **600 W server‑edition card** (worst case):

* ΔT = 10 °C:  
  CFM ≈ 600 × 3.16 / 10 ≈ **190 CFM through the card**

* ΔT = 15 °C:  
  CFM ≈ 600 × 3.16 / 15 ≈ **125 CFM through the card**

For a **300 W card**:

* ΔT = 10 °C → ≈ 95 CFM  
* ΔT = 15 °C → ≈ 63 CFM  

These numbers are **per‑card airflow through the heatsink**, not just total chassis CFM. In a 2U/4U server, this is usually achieved by **4–8 high‑pressure 40 mm or 60 mm fans in parallel**.

Because these are rough calculations, the practical takeaway:

> Aim for **~100–200 CFM of *directed, high‑pressure airflow per passive RTX 6000*** along the card length, depending on whether it’s a 300 W Ada or 600 W Blackwell server model and your inlet temperature.

You will want to cross‑check with whatever server platform you’re using (Dell or otherwise).

---

### 3. Fan *type* to use

Given this is the **server/passive edition**, you should think in terms of **server fan modules**, not consumer “case fans”.

#### Key characteristics you need

* **High static pressure**, not just high free‑air CFM  
  You’re pushing air through:
  * Dense GPU heatsink fins
  * Possibly CPU heatsinks
  * Drive cages / filters / front bezel  
  Look for fans with **≥ 5–10 mm H₂O** static pressure (many server 40×56 mm fans are >20 mm H₂O).

* **Front‑to‑back airflow** aligned with the GPU  
  The card is designed to sit in a front‑to‑back tunnel. Random side intakes or top exhausts won’t cut it unless you build ducting.

* **Redundancy and monitoring**  
  Passive GPUs assume the chassis has **multiple hot‑swap fans** plus tach monitoring and automatic thermal throttling / shutdown.

#### Practical recommendations

**If you are in a Dell PowerEdge server (strongly preferred):**

* Use a **Dell GPU‑qualified platform** and install the **highest‑performance system fan kit** that’s supported for that configuration (for example, “High Performance Fans” in R750xa/R760xa, XE9680‑class platforms, etc.).  
* Dell’s qualified GPU servers (e.g., PowerEdge XE/XE9680, R760xa, older R750xa) are designed for 300–700 W accelerators and will meet NVIDIA’s airflow requirements without you having to compute CFM explicitly.

Since you’re at Dell ISG, the cleanest and safest path is:

* Put the card into a **Dell platform listed on NVIDIA’s “NVIDIA‑Certified Systems” list** for RTX 6000 / RTX PRO 6000 Server Edition, or  
* Verify with internal thermal/engineering docs for the target chassis that it supports **passive 300–600 W PCIe GPUs** with the right fan kit.

**If you’re trying to cool it in a non‑standard chassis (e.g., lab test setup, generic 4U, workstation case):**

* Use **multiple 120 mm or 92 mm fans with high static pressure** (server‑grade or Noctua industrialPPC / Delta / Sunon).
* Build/3D‑print **baffles or ducts** so that air is forced:
  * In at the **front of the card**
  * Along the fins
  * Out the back  
  This is exactly what others are doing when putting server‑edition cards into workstations: they add custom ducts to ensure the card sees true front‑to‑back flow rather than just turbulent mixing.

* Target, at minimum:
  * For a ~300 W card: at least **2× 120 mm high‑pressure fans** at medium RPM aimed directly at the card  
  * For a 600 W card: **3–4× 120 mm high‑pressure fans** or equivalent, plus ducting

---

### 4. Concrete fan model examples (non‑Dell)

These are representative **non‑Dell** options commonly used in lab / custom server builds for high‑TDP GPUs:

* **Noctua NF‑F12 industrialPPC‑3000 PWM**  
  * Size: 120 mm  
  * Max ~110 CFM free air, high static pressure (~7.6 mm H₂O)  
  * Very robust, good for sustained high speed

* **Delta Electronics AFB / PFR / PFC series** 120 mm or 92 mm server fans  
  * Often 100–150 CFM with **very high static pressure** (>10 mm H₂O)  
  * Extremely loud but ideal if you’re building a true “server‑like” airflow channel

* **San Ace (Sanyo Denki) 9G or 9H series**  
  * Also widely used in datacenter gear for high‑pressure cooling

With these, the absolutely critical part is **ducting**: even 150 CFM fans won’t help if most of the air bypasses the GPU.

---

### 5. Example reference systems (for airflow expectations)

You can look at what other vendors do with the **RTX PRO 6000 Blackwell Server Edition**:

* **Supermicro** supports up to **8 RTX PRO 6000 Blackwell Server Edition** GPUs in certain 4U systems, explicitly optimized for “maximum thermal efficiency and airflow” with high‑performance fan walls.  

* **Lenovo ThinkSystem** has a dedicated product guide for the RTX PRO 6000 Blackwell Server Edition PCIe GPU, describing it as a **600 W, full‑height, full‑length, passive** card, intended for validated ThinkSystem servers.

These systems give you a sense of the class of cooling solution expected: **fan walls, baffles, and fully engineered airflow channels**, not ad‑hoc case fans.

---

### 6. Actionable steps for your build

Given your situation (“RTX 6000 Server edition (fanless) in a server”):

1. **Identify exact card**  
   * Is it **RTX 6000 Ada passive** (~300 W) or **RTX PRO 6000 Blackwell Server Edition** (up to 600 W)?  
   * Check PNY/NVIDIA part number; Blackwell server cards explicitly state “Server Edition” and 600 W board power.

2. **Check target server platform**  
   * If it’s a **Dell PowerEdge**, confirm:
     * The chassis is **GPU‑certified for passive PCIe GPUs** at your board power  
     * You’re using the **High‑Performance Fan Kit**  
   * If not listed/validated, you’re in “best‑effort” territory and should instrument temps carefully.

3. **Design airflow**  
   * Ensure **front‑to‑back** airflow across the card  
   * If necessary, **3D‑print or fabricate baffles** to force air through the heatsink (like the Lenovo P‑series workstation experiments).

4. **Measure & validate**  
   * Use `nvidia-smi --query-gpu=temperature.gpu --format=csv` under worst‑case load  
   * Keep GPU temperature **well below the throttle point** (typically around 85 °C for pro cards; exact limit in NVIDIA docs)  
   * If temps are high or unstable, increase fan speed, add more fans, or improve ducting.

---

### 7. Dell & non‑Dell sources you can cite internally / share

**Non‑Dell / public sources:**

* Central Computer overview of RTX PRO 6000 Blackwell Server Edition (passive, up to 600 W, requires external airflow):  
  [https://www.centralcomputer.com/blog/post/understanding-the-nvidia-rtx-6000-pro-blackwell-lineup-workstation-max-q-and-server-editions](https://www.centralcomputer.com/blog/post/understanding-the-nvidia-rtx-6000-pro-blackwell-lineup-workstation-max-q-and-server-editions)

* VAST AI article contrasting RTX 6000 variants and explaining the server edition’s need for chassis airflow:  
  [https://vast.ai/article/which-nvidia-rtx-6000-is-right-for-you](https://vast.ai/article/which-nvidia-rtx-6000-is-right-for-you)

* Supermicro systems with RTX PRO 6000 Blackwell Server Edition (for example server designs and airflow assumptions):  
  [https://www.supermicro.com/en/featured/rtx-pro-6000-systems](https://www.supermicro.com/en/featured/rtx-pro-6000-systems)

* Lenovo ThinkSystem RTX PRO 6000 Blackwell Server Edition product guide (600 W, passive, datacenter platform):  
  [https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu)

* PNY RTX PRO 6000 Blackwell Server Edition product page:  
  [https://www.pny.com/nvidia-rtx-pro-6000-blackwell](https://www.pny.com/nvidia-rtx-pro-6000-blackwell)

* General explanation of passive high‑power GPUs and the chassis‑airflow requirement:  
  [https://www.servethehome.com/pny-shows-passive-nvidia-quadro-rtx-6000-and-rtx-8000-versions/](https://www.servethehome.com/pny-shows-passive-nvidia-quadro-rtx-6000-and-rtx-8000-versions/)

---

If you tell me which **exact RTX 6000 variant** you have (Ada vs PRO 6000 Blackwell Server Edition) and which **Dell server model** you’re targeting, I can be more specific about whether the platform can safely cool it and what fan kit / airflow configuration you should plan for.