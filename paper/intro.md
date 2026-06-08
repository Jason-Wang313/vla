# Introduction

VLA models are attractive because they connect language, vision, and action. A user can say what should happen, a robot observes the scene, and the policy proposes an action or trajectory.

Deployment, however, often needs choosing among candidate actions, not merely producing one plausible language-conditioned action. Increasing `N` increases selection pressure. If the scorer ranks semantic plausibility above physical feasibility in the upper tail, Best-of-N can amplify small semantic/physical ranking errors into high-N failures.

This paper argues that VLA action selection needs selected-tail physical utility diagnostics, not only average imitation quality or language plausibility scores. We provide an exact finite selection law, VLA-specific diagnostics, controlled and learned VLA-style evidence, and repair methods based on physical verification and calibration.
