# Process Constraint Variants

These configs are generated for testing process-only constraints against Model 3.
They keep `use_counterfactual_constraint: false` and run with `--only-model3`.

The process loss constrains the final target prediction response to a perturbation
of a variable, so the rules below intentionally focus on plausible monotonic
effects on final concentrate TFe, not local equipment relations.
