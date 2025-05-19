# B25ITK57 Airhockey AI

Dette prosjektet inneholder et simuleringsmiljø og et sett med Reinforcement Learning-agenter for å trene og evaluere AI-strategier i et Air Hockey-spill. Prosjektet er utviklet i forbindelse med en bacheloroppgave ved Høgskolen i Østfold.

Innhold:

- Simulert Air Hockey-miljø basert på MuJoCo
    
- Egenimplementert RL-agent

- PPO-agent (Proximal Policy Optimization) med støtte for fasebasert reward shaping

- Regelbasert agent for baseline-evaluering

- Logging og visualisering med TensorBoard og matplotlib

Mapper og viktige filer

- environment/ — Simulasjonsmiljø, wrappers og miljødefinisjoner

- environment/env_settings/environments/data/ — XML-filer for MuJoCo-scenen (table.xml)

- environment/neural_network/ — Egenimplementert RL-agent
    
- environment/PPO_training/ — PPO-agent, rewards, trainer og treningsskript
    
- rule_based_ai_scripts/ — Regelbasert motstander

- ui/ — Scripts for å kjøre simulasjon med brukergrensesnitt(sim-to-real)

# Bruks Instruksjoner:

Installer alle nødvendige Python-pakker med:
    
    pip install -r requirements.txt

PPO-trening: start PPO-treningen med
    
    python environment/PPO_training/run_ppo_training.py

Juster eller endre belønningsfunksjoner for PPO-agenten:
    
    environment/PPO_training/ppo_rewards.py for å tilpasse hvordan agenten får poeng.
    
Juster n_intermediate_steps=1 til n_intermediate_steps=20 i env_base for PPO trening:
    
    env_base > n_intermediate_steps=20

Tren egendefinert RL-agent: Tren den egendefinerte agenten med
    
    python environment/neural_network/training_step.py
    
For å endre belønningsfunksjonen for denne agenten, gjør justeringer i:
    
    environment/env_settings/environments/iiwas/env_base.py (se reward()-metoden).

Regelbasert agent mot regelbasert agent:
    
    Kjør main.py


Tips:

    Sørg for at alle MuJoCo-XML-filer ligger i environment/env_settings/environments/data/.

Bruk TensorBoard for å følge treningen ved å kjøre
    
    tensorboard --logdir tensorboard_logs/
