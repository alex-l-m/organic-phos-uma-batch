# Compare experimental phosphorescence emmission energies to computed singlet-triplet gaps from UMA
library(tidyverse)
library(cowplot)
library(ggrepel)
library(tidymodels)

theme_set(theme_cowplot() + theme(plot.background = element_rect(fill = 'white')))

smiles_energy_uma <- read_csv('smiles_energy_uma.csv', col_types = cols( 
    mol_id = col_character(), 
    smiles = col_character(), 
    absorption_energy = col_double(), 
    emission_energy = col_double(), 
    uma_st = col_double() 
)) 

comparison_plt <- smiles_energy_uma |>
    ggplot(aes(y = uma_st, x = emission_energy, label = mol_id)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, linetype = 'dashed', color = 'red') +
    geom_smooth(method = 'lm', se = FALSE, color = 'blue') +
    geom_text_repel() +
    coord_obs_pred()

ggsave('comparison_plot.png', comparison_plt, width = unit(6, 'in'), height = unit(6, 'in'), dpi = 300)
