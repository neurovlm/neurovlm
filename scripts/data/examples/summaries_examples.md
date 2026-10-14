# PubMed summaries: re-creation examples

Written by `scripts/data/09_summaries.py`. The summary prompt (`docs/02_data/07_transform_text.ipynb`, cell 3) is the
system message and the paper's title and abstract, joined by `[SEP]`, the user message. Decoding samples, so each
run gives a new wording. Each new summary is shown next to the two released summaries of the same test-set paper,
`neuro_summaries.parquet` (neurovlm/neuro_image_papers) and `pubmed_summary_qa.parquet` (neurovlm/pubmed_summary_qa,
whose card names Llama-3.3-70B-Instruct), with their MiniLM cosine.

Mean cosine to neuro_summaries: 0.91 (same paper) against 0.39 (other papers of this set).
Mean cosine to pubmed_summary_qa: 0.83. Mean length: new 180 words,
neuro_summaries 128, pubmed_summary_qa 59.

Settings: `{"model": "meta-llama/Llama-3.3-70B-Instruct", "provider": "hf-router", "router": "https://router.huggingface.co/v1", "temperature": 0.3, "top_p": 0.9, "top_k": 50, "max_tokens": 256}`

## PMID 15820711: Brain activation of patients with obsessive-compulsive disorder during neuropsychological and symptom provocation tasks before and after symptom improvement: a functional magnetic resonance imaging st

Reading: approximately similar: same topic, different wording (cosine 0.90 to neuro_summaries, 0.80 to pubmed_summary_qa).

**New summary**

Obsessive-compulsive disorder (OCD) is a complex neuropsychiatric condition characterized by hyperactivity in the frontal cortex, which has been implicated in the pathophysiology of the disorder. Research has shown that OCD is associated with abnormal brain activity in regions such as the orbitofrontal, dorsolateral-prefrontal, and anterior cingulate cortices, which are involved in executive control, emotion regulation, and error detection. Studies have also suggested that changes in brain activity in these regions can influence cognitive processes, including working memory and attention. As symptoms of OCD improve, either through pharmacological or behavioral interventions, there tends to be a decrease in hyperactivation of the frontal lobe in response to symptom-provoking stimuli, and an increase in activity in posterior brain regions, such as the parietal cortex and cerebellum, which are involved in action monitoring and cognitive control. This shift in brain activity patterns is thought to reflect changes in the neural circuits underlying OCD, and may be related to improvements in cognitive function and symptom reduction. Overall, the neural mechanisms underlying OCD involve a complex interplay between frontal and posterior brain regions, and understanding these relationships is essential for developing effective treatments for the disorder.

**Released, neuro_summaries.parquet**

> In individuals with obsessive-compulsive disorder (OCD), hyperactivity of the frontal cortex has been implicated in the pathophysiology of the disorder. However, the relationship between abnormal brain activity, clinical improvement, and neuropsychological function in OCD remains unclear. Research has shown that in patients with OCD, certain brain regions, including the orbitofrontal, dorsolateral-prefrontal, and anterior cingulate cortices, exhibit increased activation in response to symptom-provocative tasks. After treatment and symptomatic improvement, this hyperactivation decreases. Conversely, activation in posterior brain regions, such as the parietal cortex and cerebellum, increases in response to tasks that require action-monitoring function, such as the Stroop task. These findings suggest that changes in brain activity patterns may be associated with clinical improvement in OCD.

**Released, pubmed_summary_qa.parquet**

> Obsessive-compulsive disorder (OCD) is characterized by hyperactivity in the frontal cortex. The relationship between brain activity, clinical improvement, and neuropsychological function in OCD is not well understood. Functional magnetic resonance imaging (fMRI) studies have investigated changes in brain function in patients with OCD during cognitive and symptom provocation tasks. OCD is a mental health disorder marked by recurring, intrusive thoughts and repetitive behaviors.

## PMID 19745020: BDNF val66met polymorphism influences motor system function in the human brain.

Reading: approximately similar: same topic, different wording (cosine 0.91 to neuro_summaries, 0.93 to pubmed_summary_qa).

**New summary**

The brain-derived neurotrophic factor (BDNF) val66met polymorphism has been associated with changes in brain function, particularly in the motor system. This polymorphism affects the release of BDNF, a growth factor crucial for brain plasticity and repair, which can influence motor learning and short-term plasticity. Research has shown that variations in BDNF can impact the function of sensorimotor networks, leading to differences in brain activation patterns and motor performance. Studies have found that individuals with the val66met polymorphism tend to exhibit altered brain activity, including reduced activation volume in certain brain regions, which can affect motor learning and retention. The BDNF val66met polymorphism has been linked to impaired motor system function, characterized by decreased plasticity and increased error during short-term learning, highlighting the importance of BDNF in motor cognition. Overall, the val66met polymorphism is a significant factor in understanding individual differences in motor system function and learning, with potential implications for the development of novel therapeutic strategies.

**Released, neuro_summaries.parquet**

> The brain-derived neurotrophic factor (BDNF) plays a crucial role in brain functions such as plasticity and repair. A common polymorphism, val(66)met, affects the activity-dependent release of BDNF, leading to differences in brain motor system function. In the human brain, this polymorphism influences the activation of a broad sensorimotor network, particularly in regions involved in motor control, such as the brain's motor cortex and basal ganglia. Individuals with the polymorphism exhibit reduced activation in these areas, which can impact short-term plasticity and learning. As a result, they may experience greater difficulty with motor learning and retention, particularly in tasks that require rapid adaptation, such as driving. This polymorphism is associated with altered brain motor system function and impaired motor learning abilities.

**Released, pubmed_summary_qa.parquet**

> Brain-derived neurotrophic factor (BDNF) is crucial for brain plasticity and repair. The val66met polymorphism affects BDNF release and is associated with differences in motor system function, short-term plasticity, and learning. This polymorphism influences brain activation and motor learning, with potential implications for understanding brain function and behavior.

## PMID 21172325: I remember you: a role for memory in social cognition and the functional neuroanatomy of their interaction.

Reading: approximately similar: same topic, different wording (cosine 0.92 to neuro_summaries, 0.77 to pubmed_summary_qa).

**New summary**

The interaction between memory and social cognition is a complex phenomenon that has been explored in the context of autobiographical memory and mentalizing, which is the ability to infer the thoughts and feelings of others. Research has shown that these processes share a common neural substrate, involving brain regions such as the hippocampus, temporal parietal junction, and prefrontal cortex. The overlap between these two processes is thought to facilitate the integration of personal and interpersonal information, allowing individuals to draw upon past experiences to inform their understanding of others and guide social behavior. Studies have consistently found that brain regions involved in autobiographical memory and mentalizing, such as the superior temporal sulcus and posterior cingulate cortex, exhibit functional connectivity and overlap in their activity patterns. This functional overlap is believed to enable the transformation of personal experiences into social conceptual knowledge, which can in turn support strategic social behavior and goal-directed actions. Overall, the relationship between memory and social cognition is a critical area of research, highlighting the importance of memory in shaping our understanding of others and navigating complex social situations.

**Released, neuro_summaries.parquet**

> During social cognition, remembering personal events (autobiographical memory) and inferring others' thoughts and feelings (mentalizing) share a common neural basis. This overlap is supported by the functional neuroanatomy of brain regions involved in both processes, including the left hippocampus, which plays a key role in memory, and the right temporal parietal junction, which is crucial for mentalizing. When these regions are active, they are correlated with a distributed network of brain areas, including the superior temporal sulcus, anterior temporal lobe, and dorsomedial prefrontal cortex, among others. This network facilitates the integration of personal and interpersonal information, allowing personal experiences to become social conceptual knowledge. This knowledge, in turn, informs strategic social behavior in support of personal goals. The integration of memory and mentalizing is essential for effective social cognition, highlighting the importance of memory in this process.

**Released, pubmed_summary_qa.parquet**

> Autobiographical memory, the recollection of personal past events, and mentalizing, the inference of others' thoughts and feelings, share a common neural basis. The default mode network, including regions such as the hippocampus and temporal parietal junction, is involved in both processes. This overlap enables the integration of personal and interpersonal information, informing social behavior and goal-oriented decision-making.

## PMID 23140909: Quantitative cerebral blood flow mapping and functional connectivity of postherpetic neuralgia pain: a perfusion fMRI study.

Reading: approximately similar: same topic, different wording (cosine 0.89 to neuro_summaries, 0.81 to pubmed_summary_qa).

**New summary**

Postherpetic neuralgia (PHN) is a chronic pain condition that has been associated with changes in brain activity and connectivity, particularly in regions involved in pain processing, emotion regulation, and sensory perception. Research in this area has focused on understanding the neural mechanisms underlying PHN pain, including the role of the default mode network, salience network, and reward circuitry. Studies have shown that PHN can influence functional connectivity between brain regions, such as the striatum, thalamus, insula, and amygdala, which are involved in pain modulation, emotional processing, and sensory integration. Increased activity in these regions, particularly in the somatosensory cortex and insula, has been linked to pain intensity, suggesting a complex interplay between sensory, emotional, and cognitive processes. The condition has also been associated with altered cerebral blood flow patterns, with increases in regions involved in pain processing and decreases in areas related to executive control and cognitive regulation. Overall, investigating the neural correlates of PHN pain can provide valuable insights into the central mechanisms underlying this condition and inform the development of novel therapeutic strategies.

**Released, neuro_summaries.parquet**

> Postherpetic neuralgia (PHN) pain is associated with altered resting-state brain activity, particularly in regions involved in pain processing and emotion regulation. In individuals with PHN, increased cerebral blood flow (CBF) is observed in key brain regions, including the left striatum, right thalamus, left primary somatosensory cortex, left insula, left amygdala, and left inferior parietal lobule. Conversely, decreased CBF is found in the frontal cortex. The CBF in certain regions, such as the left caudate, left insula, and left primary somatosensory cortex, is highly correlated with self-reported pain intensity, suggesting a link between these areas and PHN pain perception. Functional connectivity analysis reveals that the reward circuitry and circuitry among the striatum, thalamus, and insula are highly correlated with PHN pain, indicating potential neural networks involved in the condition. These findings suggest that noninvasive brain perfusion imaging can provide insights into the central mechanisms underlying PHN pain.

**Released, pubmed_summary_qa.parquet**

> Postherpetic neuralgia (PHN) is a condition characterized by chronic pain following herpes zoster infection. Resting-state brain activity in PHN patients shows altered cerebral blood flow (CBF) patterns compared to healthy individuals. Regions such as the striatum, thalamus, and primary somatosensory cortex exhibit changes in CBF, which are correlated with pain intensity. Functional connectivity between these regions is also affected in PHN patients.

## PMID 23322403: Neurocognitive Brain Response to Transient Impairment of Wernicke's Area.

Reading: approximately similar: same topic, different wording (cosine 0.90 to neuro_summaries, 0.79 to pubmed_summary_qa).

**New summary**

The topic of neurocognitive brain response to transient impairment of Wernicke's Area is concerned with understanding how the brain adapts and compensates for damage to key language processing centers. Research in this area focuses on the cognitive processes of language comprehension and the brain systems involved, including the left posterior superior temporal gyrus (LSTGp) and its role in the primary language network. Studies have shown that when this region is impaired, the brain can reconfigure its information processing capabilities to maintain cognitive performance, often through increased synchronization between compensatory regions and decreased synchronization within the primary language network. This adaptation can involve the recruitment of additional brain areas, such as the contralateral homolog of the impaired region, adjacent areas, and regions involved in discourse monitoring, like the medial frontal gyrus. Overall, this line of research has implications for understanding how the brain responds to trauma, including traumatic brain injury, stroke, and seizure, and can inform the development of new treatments and therapies. By exploring the neural mechanisms of adaptation and compensation, researchers can gain insights into the complex interactions between different brain regions and networks that underlie language processing and other cognitive functions.

**Released, neuro_summaries.parquet**

> When Wernicke's Area, a key region in the left posterior superior temporal gyrus, is temporarily impaired, the brain adapts to maintain cognitive performance through reconfiguration of its information processing capabilities. This adaptation involves increased synchronization between compensating brain regions, such as the contralateral homolog, adjacent areas, and the medial frontal gyrus, which is involved in discourse monitoring. In contrast, synchronization within the primary language network decreases, and activation at the impaired site and in distal regions initially decreases before recovering. This compensatory response enables the brain to maintain language comprehension capabilities despite the temporary impairment of Wernicke's Area. The brain's ability to adapt and reconfigure its networks in response to trauma has implications for understanding and treating conditions such as traumatic brain injury, stroke, and seizure.

**Released, pubmed_summary_qa.parquet**

> Wernicke's area, located in the left posterior superior temporal gyrus (LSTGp), plays a crucial role in language comprehension. When this area is impaired, the brain adapts to maintain cognitive performance by reconfiguring its information processing capabilities. This adaptation involves changes in synchronization and activation across different brain regions, including compensatory regions such as the contralateral homolog and areas involved in discourse monitoring.

## PMID 24825504: Imagining triadic interactions simultaneously activates mirror and mentalizing systems.

Reading: approximately similar: same topic, different wording (cosine 0.93 to neuro_summaries, 0.89 to pubmed_summary_qa).

**New summary**

The ability to engage in triadic interactions, which involves coordinating interactions between oneself, another person, and an external object, is a complex social cognitive process that has been associated with the activation of multiple brain systems. Research in this area suggests that triadic interactions simultaneously engage the mirror system, which is involved in empathy and understanding other's actions, and the mentalizing system, which is responsible for attributing mental states to oneself and others. The mentalizing system, including regions such as the medial prefrontal cortex and temporoparietal junction, plays a crucial role in self-referential and other-referential processing, while the mirror system, including areas such as the inferior frontal gyrus and inferior parietal lobe, is thought to be involved in simulating others' actions and intentions. Studies have shown that triadic interactions can influence the functional connectivity between these systems, leading to a more complex neural network involved in social cognition. Overall, the neural mechanisms underlying triadic interactions are thought to be distinct from those involved in purely self-referential or other-referential processing, and are associated with changes in brain activity and connectivity in key social cognitive networks.

**Released, neuro_summaries.parquet**

> When imagining triadic interactions involving oneself, another person, and an external object, the brain activates a complex network of regions. This network includes core mentalizing areas, such as the medial prefrontal cortex, posterior cingulate cortex, precuneus, and temporoparietal junction, which are involved in understanding social interactions and mental states. Additionally, triadic interactions specifically activate mirror neuron areas, including the inferior frontal gyrus and inferior parietal lobe, which are typically associated with self-other distinction and action observation. The mentalizing network shows strong functional connectivity with mirror neuron areas during triadic engagements, suggesting a coordinated processing of social interactions. This neural social cognitive network is more complex than that involved in purely self-referential or other-referential processing, highlighting the unique cognitive demands of triadic interactions.

**Released, pubmed_summary_qa.parquet**

> Triadic social interactions involve oneself, another person, and an external object. These interactions are a complex form of social cognition that engages multiple brain systems. The mentalizing system, which includes areas such as the medial prefrontal cortex, is active during triadic interactions. Additionally, mirror neuron areas, including the inferior frontal gyrus, are also activated, suggesting a link between social cognition and empathy.
