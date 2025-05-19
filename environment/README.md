B25ITK57_Airhockey

Dette prosjektet inneholder et simuleringsmiljø og et sett med Reinforcement Learning-agenter for å trene og evaluere AI-strategier i et Air Hockey-spill. Prosjektet er utviklet i forbindelse med en bacheloroppgave ved Høgskolen i Østfold.
Innhold

    Simulert Air Hockey-miljø basert på MuJoCo
    
    Egenimplementert RL-agent for sammenligning

    PPO-agent (Proximal Policy Optimization) med støtte for fasebasert reward shaping

    Regelbasert agent for baseline-evaluering

    Logging og visualisering med TensorBoard og matplotlib

Mapper og viktige filer

    environment/ — Simulasjonsmiljø, wrappers og miljødefinisjoner

    environment/env_settings/environments/data/ — XML-filer for MuJoCo-scenen (table.xml)

    environment/neural_network/ — Egenimplementert RL-agent
    
    environment/PPO_training/ — PPO-agent, rewards, trainer og treningsskript
    
    rule_based_ai_scripts/ — Regelbasert motstander

    ui/ — Scripts for å kjøre simulasjon med brukergrensesnitt(sim-to-real)
