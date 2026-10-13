"""Constants copied from scripts/figures/main/{fig3_newgen,figS4_clip_ica}.py and their outputs; update them if those
scripts change."""

# the six rows of the final Fig. 3, top to bottom (scripts/figures/main/fig3_newgen_text.json): the hand-picked
# 189, 149 and 101, 110, 92, 74, which replace 124, 152, 147, 167 because those broke the pool-tercile rule (global map
# calibration; canonical titles after the tags the model writes)
KEEP_TEXT_IDXS = [189, 149, 101, 110, 92, 74]

# from scripts/figures/main/figS4_clip_ica.py
COHORTS = (("UKBICA", "UK Biobank"), ("HCPICA", "Human Connectome Project"))

NETS = {
    "frontoparietal": (
        "A domain-general control network supporting flexible, goal-directed behavior (task set maintenance, rule switching, "
        "working-memory control, adaptive decision-making). Neuroanatomically centered on dorsolateral prefrontal cortex "
        "(middle frontal gyrus), inferior/superior parietal lobule and intraparietal sulcus, with contributions from dorsal "
        "premotor areas and midline control regions."
    ),
    "language": (
        "A left-lateralized network for speech perception/production and sentence-level comprehension (phonology, syntax, "
        "semantics, speech planning). Typically includes inferior frontal gyrus (Broca’s area: pars opercularis/triangularis), "
        "posterior superior temporal gyrus/sulcus, middle temporal gyrus, anterior temporal lobe, and angular gyrus, with white-matter "
        "support via arcuate/superior longitudinal fasciculus."
    ),
    "memory": (
        "A network supporting episodic memory encoding and retrieval (scene/context construction, recollection, consolidation). "
        "Anchored in the medial temporal lobe (hippocampus, parahippocampal/entorhinal cortex) and strongly coupled to posterior "
        "cingulate/precuneus, retrosplenial cortex, medial prefrontal cortex, and lateral parietal cortex."
    ),
    "motor": (
        "A network for planning, initiation, and execution of voluntary movements (motor sequencing, force control). Core nodes "
        "include primary motor cortex (precentral gyrus), supplementary motor area/pre-SMA, dorsal premotor cortex, with subcortical "
        "loops through basal ganglia and cerebellum."
    ),
    "action-observation": (
        "Often overlapping with the ‘mirror’ system, this network supports action understanding, imitation, and mapping observed actions "
        "to motor representations. Involves inferior frontal gyrus/ventral premotor cortex, inferior parietal lobule (supramarginal/angular), "
        "and posterior superior temporal sulcus as a key visual–biological motion hub."
    ),
    "reward": (
        "A network supporting valuation, reinforcement learning, and motivated behavior (prediction error, incentive salience, reward-guided choice). "
        "Includes ventral striatum/nucleus accumbens, ventromedial/orbitofrontal cortex, anterior cingulate cortex, amygdala, hippocampus, and midbrain "
        "dopaminergic regions (VTA/substantia nigra) with thalamic interactions."
    ),
    "semantic": (
        "A network for conceptual knowledge and meaning integration across modalities (word meaning, category knowledge, semantic control). "
        "Often includes anterior temporal lobe (semantic hub), middle/inferior temporal gyrus, angular gyrus, posterior superior temporal cortex, "
        "and inferior frontal gyrus for controlled semantic retrieval/selection."
    ),
    "sensorimotor": (
        "A network integrating somatosensory input with motor output (tactile perception, proprioception, sensorimotor coordination). "
        "Centered on precentral and postcentral gyri, paracentral lobule, SMA, and secondary somatosensory cortex (parietal operculum), "
        "with contributions from insula and cerebellum for integration and timing."
    ),
    "visual": (
        "A set of occipital and occipito-temporal systems supporting visual perception (low-level feature processing through higher-level object/scene processing). "
        "Includes primary visual cortex (calcarine sulcus/V1), extrastriate areas (V2–V5/MT), lateral occipital cortex, fusiform gyrus, and parahippocampal "
        "place area for scenes, often extending into intraparietal regions for visuospatial processing."
    ),
    "central executive": (
        "A classic control network for working memory, sustained attention, and top-down regulation of thought/action. Neuroanatomically overlaps strongly "
        "with frontoparietal control systems, emphasizing dorsolateral prefrontal cortex and posterior parietal cortex (IPS/SPL), with coordination via dorsal "
        "ACC/pre-SMA in demanding tasks."
    ),
    "cognitive control": (
        "A network implementing top-down control over perception and action (conflict monitoring, inhibition, rule implementation, error processing). "
        "Typically spans dorsolateral and ventrolateral prefrontal cortex, dorsal anterior cingulate cortex/pre-SMA, and posterior parietal cortex, "
        "with basal ganglia contributions for gating and action selection."
    ),
    "default mode": (
        "The default mode network supports internally oriented cognition (mind-wandering, autobiographical memory, self-referential thought, future simulation, "
        "social cognition). Core hubs include posterior cingulate cortex/precuneus and medial prefrontal cortex, with lateral parietal cortex (angular gyrus) "
        "and medial temporal lobe coupling (hippocampus/parahippocampal) depending on memory demands."
    ),
    "executive control": (
        "A network for executive functions like planning, set shifting, maintaining goals, and suppressing distractions. Neuroanatomically overlaps with "
        "frontoparietal/central executive systems (dlPFC and posterior parietal), often coordinating with dorsal ACC/pre-SMA for performance monitoring "
        "and with basal ganglia for action/response selection."
    ),
    "salience": (
        "A network that detects behaviorally relevant stimuli and coordinates switching between internal (DMN) and externally oriented control/attention systems "
        "(rapid orienting, interoception, threat detection). Anchored in anterior insula (especially right), dorsal anterior cingulate cortex, and subcortical "
        "nodes including amygdala, thalamus, and periaqueductal/midbrain regions."
    ),
    "attention": (
        "A broad term for systems supporting selective attention and attentional control (orienting, prioritizing sensory inputs, maintaining vigilance). "
        "Typically spans frontoparietal regions—frontal eye fields, intraparietal sulcus/superior parietal lobule—and interacts with ventral fronto-insular "
        "and temporoparietal regions depending on goal-driven vs. stimulus-driven demands."
    ),
    "ventral attention": (
        "A right-lateralized network for stimulus-driven reorienting to salient or unexpected events (circuit breaker for attention, target detection). "
        "Key nodes include temporoparietal junction, ventral frontal cortex (inferior/middle frontal gyrus), and anterior insula, interacting with sensory "
        "cortex to shift attention rapidly."
    ),
    "dorsal attention": (
        "A bilateral network for goal-directed, top-down allocation of attention (spatial orienting, sustained visuospatial attention, saccade planning). "
        "Centered on intraparietal sulcus/superior parietal lobule and frontal eye fields, with contributions from dorsal premotor regions and visual cortex "
        "for prioritized processing."
    ),
}
