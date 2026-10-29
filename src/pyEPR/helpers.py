from device import KeysightE36231A, MokuGoReadOnly


def power_supply_sanity_check(moku: MokuGoReadOnly, keysight: KeysightE36231A):
    temp = keysight.magnet_temp()
    if temp > 35:
        print(f"Magnet temperature: {temp} C")
    ps = moku.get_power_supply([1, 2, 3, 4])
    curr1 = ps[1]["actual_current"]
    curr2 = ps[2]["actual_current"]
    curr3 = ps[3]["actual_current"]
    curr4 = ps[4]["actual_current"]
    if curr1 > -0.005:
        print(f"Current on supply 1 too low: {curr1:.4f} A")
    if curr2 < 0.050:
        print(f"Current on supply 2 too low: {curr2:.4f} A")
    if curr3 < 0.020:
        print(f"Current on supply 3 too low: {curr3:.4f} A")
    if curr4 < 0.005:
        print(f"Current on supply 4 too low: {curr4:.4f} A")
