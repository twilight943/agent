from src.services.item_service import ItemService


def item_lookup_tool(service: ItemService, item_id: int) -> dict[str, object] | None:
    return service.get_item(item_id)
