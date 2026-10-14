# PubMed summaries: re-creation examples

Written by `scripts/data/09_summaries.py`. The summary prompt (`docs/02_data/07_transform_text.ipynb`, cell 3) is the
system message and the paper's title and abstract, joined by `[SEP]`, the user message. Decoding samples, so each
run gives a new wording. Each new summary is shown next to the two released summaries of the same test-set paper,
`neuro_summaries.parquet` (neurovlm/neuro_image_papers) and `pubmed_summary_qa.parquet` (neurovlm/pubmed_summary_qa),
with their MiniLM cosine and a format check (paragraphs, words, digits, non-Latin characters).

Mean cosine to neuro_summaries: 0.88 (same paper) against 0.39 (other papers of this set).
Mean cosine to pubmed_summary_qa: 0.79. Mean length: new 153 words,
neuro_summaries 128, pubmed_summary_qa 59.

Settings: `{"model": "Qwen/Qwen3.5-27B", "provider": "hf-router", "router": "https://router.huggingface.co/v1", "temperature": 0.3, "top_p": 0.9, "top_k": 50, "max_tokens": 256, "enable_thinking": false}`

## PMID 15820711: Brain activation of patients with obsessive-compulsive disorder during neuropsychological and symptom provocation tasks before and after symptom improvement: a functional magnetic resonance imaging st

Reading: approximately similar: same topic, different wording (cosine 0.86 to neuro_summaries, 0.77 to pubmed_summary_qa).

Check: 1 paragraph(s), 151 words, 0 digits, 0 non-Latin characters, finish reason stop.

**New summary**

Obsessive-compulsive disorder (OCD) is characterized by distinct alterations in brain function, particularly involving hyperactivity within the frontal cortex and the cortico-striato-thalamo-cortical loops. Research indicates that the pathophysiology of OCD is closely linked to dysregulated activity in the orbitofrontal cortex, dorsolateral prefrontal cortex, and anterior cingulate cortex during states of symptom provocation, reflecting deficits in emotion regulation and error monitoring. Conversely, cognitive processes such as executive control and action monitoring, which engage parietal and cerebellar regions, often show compensatory or distinct activation patterns that may be impaired in the untreated state. Clinical interventions, including pharmacotherapy and behavioral therapy, have been associated with a normalization of these neural circuits, specifically reducing pathological frontal hyperactivation while enhancing posterior brain activity related to cognitive control. These findings suggest that symptom improvement in OCD corresponds to a rebalancing of functional connectivity between frontal regions involved in obsessive thoughts and posterior networks responsible for adaptive behavioral regulation.

**Released, neuro_summaries.parquet**

> In individuals with obsessive-compulsive disorder (OCD), hyperactivity of the frontal cortex has been implicated in the pathophysiology of the disorder. However, the relationship between abnormal brain activity, clinical improvement, and neuropsychological function in OCD remains unclear. Research has shown that in patients with OCD, certain brain regions, including the orbitofrontal, dorsolateral-prefrontal, and anterior cingulate cortices, exhibit increased activation in response to symptom-provocative tasks. After treatment and symptomatic improvement, this hyperactivation decreases. Conversely, activation in posterior brain regions, such as the parietal cortex and cerebellum, increases in response to tasks that require action-monitoring function, such as the Stroop task. These findings suggest that changes in brain activity patterns may be associated with clinical improvement in OCD.

**Released, pubmed_summary_qa.parquet**

> Obsessive-compulsive disorder (OCD) is characterized by hyperactivity in the frontal cortex. The relationship between brain activity, clinical improvement, and neuropsychological function in OCD is not well understood. Functional magnetic resonance imaging (fMRI) studies have investigated changes in brain function in patients with OCD during cognitive and symptom provocation tasks. OCD is a mental health disorder marked by recurring, intrusive thoughts and repetitive behaviors.

## PMID 19745020: BDNF val66met polymorphism influences motor system function in the human brain.

Reading: approximately similar: same topic, different wording (cosine 0.86 to neuro_summaries, 0.83 to pubmed_summary_qa).

Check: 1 paragraph(s), 149 words, 2 digits, 0 non-Latin characters, finish reason stop.

**New summary**

The BDNF val66met polymorphism represents a significant genetic factor influencing the efficiency of motor system function and neuroplasticity in the human brain. This single nucleotide variation affects activity-dependent release of brain-derived neurotrophic factor, a protein critical for synaptic plasticity and neuronal repair, thereby modulating the brain's capacity for motor learning and adaptation. Research in this area suggests that individuals carrying the met allele often exhibit altered functional responses within the sensorimotor network, including differences in activation magnitude and the ability to reorganize neural circuits following practice. These genetic differences are frequently associated with distinct patterns of short-term plasticity, where the typical expansion or refinement of motor-related brain activity may be attenuated or altered compared to non-carriers. Consequently, the presence of this polymorphism has been linked to measurable variations in motor skill acquisition and retention, highlighting the role of genetic variability in shaping the neural mechanisms underlying human motor behavior.

**Released, neuro_summaries.parquet**

> The brain-derived neurotrophic factor (BDNF) plays a crucial role in brain functions such as plasticity and repair. A common polymorphism, val(66)met, affects the activity-dependent release of BDNF, leading to differences in brain motor system function. In the human brain, this polymorphism influences the activation of a broad sensorimotor network, particularly in regions involved in motor control, such as the brain's motor cortex and basal ganglia. Individuals with the polymorphism exhibit reduced activation in these areas, which can impact short-term plasticity and learning. As a result, they may experience greater difficulty with motor learning and retention, particularly in tasks that require rapid adaptation, such as driving. This polymorphism is associated with altered brain motor system function and impaired motor learning abilities.

**Released, pubmed_summary_qa.parquet**

> Brain-derived neurotrophic factor (BDNF) is crucial for brain plasticity and repair. The val66met polymorphism affects BDNF release and is associated with differences in motor system function, short-term plasticity, and learning. This polymorphism influences brain activation and motor learning, with potential implications for understanding brain function and behavior.

## PMID 21172325: I remember you: a role for memory in social cognition and the functional neuroanatomy of their interaction.

Reading: approximately similar: same topic, different wording (cosine 0.89 to neuro_summaries, 0.81 to pubmed_summary_qa).

Check: 1 paragraph(s), 148 words, 0 digits, 0 non-Latin characters, finish reason stop.

**New summary**

The intersection of autobiographical memory and social cognition, particularly mentalizing, represents a critical area of inquiry in neuroscience, highlighting how the brain integrates personal past experiences with the understanding of others' minds. Research indicates that these distinct cognitive processes share a common functional neuroanatomy, relying heavily on a distributed network that includes the medial prefrontal cortex, posterior cingulate cortex, angular gyrus, and temporal regions. Functional connectivity studies suggest that while specific hubs like the hippocampus and the temporal parietal junction may operate independently, they both engage this broader system to facilitate the seamless integration of self-referential and interpersonal information. This neural overlap is thought to support the transformation of individual episodic memories into social conceptual knowledge, which subsequently guides strategic social behavior aligned with personal goals. Consequently, contemporary perspectives in social cognitive neuroscience increasingly emphasize the foundational role of memory systems in constructing and navigating complex social interactions.

**Released, neuro_summaries.parquet**

> During social cognition, remembering personal events (autobiographical memory) and inferring others' thoughts and feelings (mentalizing) share a common neural basis. This overlap is supported by the functional neuroanatomy of brain regions involved in both processes, including the left hippocampus, which plays a key role in memory, and the right temporal parietal junction, which is crucial for mentalizing. When these regions are active, they are correlated with a distributed network of brain areas, including the superior temporal sulcus, anterior temporal lobe, and dorsomedial prefrontal cortex, among others. This network facilitates the integration of personal and interpersonal information, allowing personal experiences to become social conceptual knowledge. This knowledge, in turn, informs strategic social behavior in support of personal goals. The integration of memory and mentalizing is essential for effective social cognition, highlighting the importance of memory in this process.

**Released, pubmed_summary_qa.parquet**

> Autobiographical memory, the recollection of personal past events, and mentalizing, the inference of others' thoughts and feelings, share a common neural basis. The default mode network, including regions such as the hippocampus and temporal parietal junction, is involved in both processes. This overlap enables the integration of personal and interpersonal information, informing social behavior and goal-oriented decision-making.

## PMID 23140909: Quantitative cerebral blood flow mapping and functional connectivity of postherpetic neuralgia pain: a perfusion fMRI study.

Reading: approximately similar: same topic, different wording (cosine 0.88 to neuro_summaries, 0.77 to pubmed_summary_qa).

Check: 1 paragraph(s), 150 words, 0 digits, 0 non-Latin characters, finish reason stop.

**New summary**

Postherpetic neuralgia (PHN) represents a chronic pain condition that induces distinct alterations in both cerebral perfusion and functional connectivity within the brain's pain-processing networks. Research in this area highlights that persistent neuropathic pain is associated with increased blood flow in key subcortical and cortical regions, including the thalamus, striatum, insula, and primary somatosensory cortex, which are integral to the sensory-discriminative and affective dimensions of pain. Conversely, these conditions often involve reduced perfusion in frontal cortical areas, potentially reflecting deficits in top-down pain modulation or executive control. Furthermore, the functional coupling between the reward circuitry, limbic structures like the amygdala, and sensory hubs appears to be significantly reorganized, suggesting that chronic pain reshapes the communication pathways underlying emotional and sensory integration. Collectively, these findings indicate that PHN is not merely a peripheral phenomenon but involves widespread central nervous system adaptations that link altered hemodynamics with the subjective experience of pain intensity.

**Released, neuro_summaries.parquet**

> Postherpetic neuralgia (PHN) pain is associated with altered resting-state brain activity, particularly in regions involved in pain processing and emotion regulation. In individuals with PHN, increased cerebral blood flow (CBF) is observed in key brain regions, including the left striatum, right thalamus, left primary somatosensory cortex, left insula, left amygdala, and left inferior parietal lobule. Conversely, decreased CBF is found in the frontal cortex. The CBF in certain regions, such as the left caudate, left insula, and left primary somatosensory cortex, is highly correlated with self-reported pain intensity, suggesting a link between these areas and PHN pain perception. Functional connectivity analysis reveals that the reward circuitry and circuitry among the striatum, thalamus, and insula are highly correlated with PHN pain, indicating potential neural networks involved in the condition. These findings suggest that noninvasive brain perfusion imaging can provide insights into the central mechanisms underlying PHN pain.

**Released, pubmed_summary_qa.parquet**

> Postherpetic neuralgia (PHN) is a condition characterized by chronic pain following herpes zoster infection. Resting-state brain activity in PHN patients shows altered cerebral blood flow (CBF) patterns compared to healthy individuals. Regions such as the striatum, thalamus, and primary somatosensory cortex exhibit changes in CBF, which are correlated with pain intensity. Functional connectivity between these regions is also affected in PHN patients.

## PMID 23322403: Neurocognitive Brain Response to Transient Impairment of Wernicke's Area.

Reading: approximately similar: same topic, different wording (cosine 0.89 to neuro_summaries, 0.80 to pubmed_summary_qa).

Check: 1 paragraph(s), 143 words, 0 digits, 0 non-Latin characters, finish reason stop.

**New summary**

Research into the neurocognitive response to transient impairment of Wernicke's area explores how the brain dynamically reconfigures its functional networks to preserve language comprehension following focal disruption. When the left posterior superior temporal gyrus, a critical hub for auditory processing and semantic integration, is compromised, the system often recruits compensatory mechanisms involving the contralateral homolog in the right hemisphere and adjacent cortical regions. This adaptive process is characterized by a shift in functional connectivity, where synchronization increases between these compensatory centers while coherence within the primary language network may temporarily decrease. Additionally, higher-order regions involved in discourse monitoring, such as the medial frontal gyrus, are frequently engaged to support executive control over the degraded linguistic input. Such network-level adaptations highlight the brain's inherent plasticity and provide a framework for understanding recovery trajectories in clinical conditions like stroke, traumatic brain injury, and seizure disorders.

**Released, neuro_summaries.parquet**

> When Wernicke's Area, a key region in the left posterior superior temporal gyrus, is temporarily impaired, the brain adapts to maintain cognitive performance through reconfiguration of its information processing capabilities. This adaptation involves increased synchronization between compensating brain regions, such as the contralateral homolog, adjacent areas, and the medial frontal gyrus, which is involved in discourse monitoring. In contrast, synchronization within the primary language network decreases, and activation at the impaired site and in distal regions initially decreases before recovering. This compensatory response enables the brain to maintain language comprehension capabilities despite the temporary impairment of Wernicke's Area. The brain's ability to adapt and reconfigure its networks in response to trauma has implications for understanding and treating conditions such as traumatic brain injury, stroke, and seizure.

**Released, pubmed_summary_qa.parquet**

> Wernicke's area, located in the left posterior superior temporal gyrus (LSTGp), plays a crucial role in language comprehension. When this area is impaired, the brain adapts to maintain cognitive performance by reconfiguring its information processing capabilities. This adaptation involves changes in synchronization and activation across different brain regions, including compensatory regions such as the contralateral homolog and areas involved in discourse monitoring.

## PMID 24825504: Imagining triadic interactions simultaneously activates mirror and mentalizing systems.

Reading: approximately similar: same topic, different wording (cosine 0.89 to neuro_summaries, 0.79 to pubmed_summary_qa).

Check: 1 paragraph(s), 175 words, 0 digits, 0 non-Latin characters, finish reason stop.

**New summary**

Triadic interactions, which involve the dynamic coordination between oneself, another individual, and an external object, represent a uniquely human social capability that engages a complex neural architecture. Research in this domain highlights the simultaneous recruitment of two distinct but interacting systems: the mentalizing network, responsible for inferring the intentions and mental states of others, and the mirror neuron system, which supports the simulation of actions and shared experiences. While self-referential and other-referential processing primarily activate core mentalizing regions such as the medial prefrontal cortex and temporoparietal junction, the specific demand of imagining a joint interaction with an object uniquely recruits classical mirror areas in the inferior frontal gyrus and inferior parietal lobe. This dual engagement suggests that understanding triadic scenarios requires an integration of cognitive perspective-taking with action simulation, a process reflected in enhanced functional connectivity between these networks. Consequently, the neural processing of such social contexts appears to rely on a more intricate interplay of brain systems than that required for simpler forms of social cognition, underscoring the specialized nature of human collaborative behavior.

**Released, neuro_summaries.parquet**

> When imagining triadic interactions involving oneself, another person, and an external object, the brain activates a complex network of regions. This network includes core mentalizing areas, such as the medial prefrontal cortex, posterior cingulate cortex, precuneus, and temporoparietal junction, which are involved in understanding social interactions and mental states. Additionally, triadic interactions specifically activate mirror neuron areas, including the inferior frontal gyrus and inferior parietal lobe, which are typically associated with self-other distinction and action observation. The mentalizing network shows strong functional connectivity with mirror neuron areas during triadic engagements, suggesting a coordinated processing of social interactions. This neural social cognitive network is more complex than that involved in purely self-referential or other-referential processing, highlighting the unique cognitive demands of triadic interactions.

**Released, pubmed_summary_qa.parquet**

> Triadic social interactions involve oneself, another person, and an external object. These interactions are a complex form of social cognition that engages multiple brain systems. The mentalizing system, which includes areas such as the medial prefrontal cortex, is active during triadic interactions. Additionally, mirror neuron areas, including the inferior frontal gyrus, are also activated, suggesting a link between social cognition and empathy.
