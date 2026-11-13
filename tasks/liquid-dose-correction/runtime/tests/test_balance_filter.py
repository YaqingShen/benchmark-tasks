import unittest,math
from native_balance import BalanceFilter


class BalanceFilterTests(unittest.TestCase):
    def test_noisy_known_loads_settle_without_changing_tare(self):
        balance=BalanceFilter()
        for mass in [120.,130.,160.]:
            for i in range(2000):balance.advance_force(mass+.15*math.sin(2*math.pi*100*i*.001),.001)
            self.assertTrue(balance.last['stable'])
            self.assertAlmostEqual(balance.last['net_g'],mass-120.,places=3)
            self.assertAlmostEqual(balance.tare_g,120.,places=3)

    def test_changing_load_is_not_stable(self):
        balance=BalanceFilter()
        for _ in range(2000):balance.advance_force(120.,.001)
        for i in range(1000):balance.advance_force(120.+i*.001,.001)
        self.assertFalse(balance.last['stable'])

    def test_removed_beaker_is_not_stable(self):
        balance=BalanceFilter()
        for _ in range(2000):balance.advance_force(120.,.001)
        for _ in range(2000):balance.advance_force(0.,.001)
        self.assertFalse(balance.last['stable'])

if __name__=='__main__':unittest.main()
