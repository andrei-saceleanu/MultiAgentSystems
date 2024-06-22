from typing import List, Dict, Any

from scipy.stats._multivariate import special_ortho_group_frozen

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage


class MyACMEAgent(HouseOwnerAgent):

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)
        self.round2percent_auction = {
            0: 1.2,
            1: 1.3,
            2: 1.4
        }
        self.round2percent_negot = {
            0: 0.5,
            1: 0.75,
            2: 1.0
        }
        self.auction_deals = {}
        self.current_negotiation = {}

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        
        res = self.round2percent_auction[auction_round] * self.budget_dict[auction_item]
        self.auction_deals[auction_item] = res

        return res
    

    def notify_auction_round_result(self, auction_item: str, auction_round: int, responding_agents: List[str]):
        print("[NOTIFICATION] ACME AUCTION ROUND RESULT")
        print(f"{auction_item} - round {auction_round} - responding agents are: {responding_agents}")        

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str, negotiation_round: int) -> float:
        print("[NOTIFICATION] NEGOTIATION OFFER")
        if negotiation_round == 0:
            self.current_negotiation = {}
            offer = self.round2percent_negot[negotiation_round] * self.auction_deals[negotiation_item]
        else:
            best_offer_so_far = None
            best_partner = None
            for k, v in self.current_negotiation.items():
                if best_offer_so_far is None:
                    best_offer_so_far = v
                    best_partner = k
                else:
                    best_partner = k if v < best_offer_so_far else best_partner
                    best_offer_so_far = min(best_offer_so_far, v)
            
            offer = min(
                self.round2percent_negot[negotiation_round] * self.auction_deals[negotiation_item],
                best_offer_so_far
            )
                    
        print(f"Offer {offer} to Company {partner_agent}")

        return offer

    def notify_partner_response(self, response_msg: NegotiationMessage) -> None:
        print("[NOTIFICATION] PARTNER RESPONSE")
        print(f"{response_msg.sender} can do {response_msg.offer}")
        self.current_negotiation[response_msg.sender] = response_msg.offer

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        print("[NOTIFICATION] NEGOTIATION WINNER")
        print(f"{winning_agent} won {negotiation_item} with {winning_offer}")
        



class MyCompanyAgent(CompanyAgent):

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)
        self.contract_wins = 0
        self.profit = {}

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        return item_budget >= self.specialties[auction_item]

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        print(f"[NOTIFICATION] WON AUCTION")
        print(f"Won the auction for {auction_item} - round {auction_round} with other {num_selected-1} companies ")


    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        print(f"[NOTIFICATION] NEGOTIATION RESPONSE")
        negotiation_item = initiator_msg.negotiation_item
        current_round = initiator_msg.round

        if self.contract_wins == 0:
            company_offer = self.specialties[negotiation_item] * (0.9**current_round)
            result = company_offer
        else:
            company_offer = self.specialties[negotiation_item] * max(1.0, 3/(current_round+1))
            result = max(company_offer, initiator_msg.offer)

        print(f"{self.role} responds to {initiator_msg.sender} with {result}")
        return result

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        print("[NOTIFICATION] WON NEGOTIATION")
        print(f"{construction_item} was assigned to {self.role} for {price}")
        self.contract_wins += 1
        self.profit[construction_item] = [price, self.specialties[construction_item]]

    def notify_negotiation_lost(self, construction_item: str) -> None:
        print("[NOTIFICATION] LOST NEGOTIATION")
        print(f"{self.role} lost the negotiation for item {construction_item}")

