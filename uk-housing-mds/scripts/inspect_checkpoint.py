import great_expectations as gx

ctx = gx.get_context(context_root_dir="great_expectations")
r = ctx.run_checkpoint(checkpoint_name="landing_all")
print("overall:", r.success)
for vr in r.list_validation_results():
    suite = vr["meta"]["expectation_suite_name"]
    print(f"\n=== {suite}: success={vr['success']} ===")
    for res in vr["results"]:
        if not res["success"]:
            cfg = res["expectation_config"]
            kw = cfg["kwargs"]
            print(f"  FAIL {cfg['expectation_type']} col={kw.get('column')}")
            result = res.get("result", {})
            print(f"    observed: {result.get('observed_value', '')}")
            pu = result.get("partial_unexpected_list", [])
            if pu:
                print(f"    unexpected sample: {pu[:5]}")
