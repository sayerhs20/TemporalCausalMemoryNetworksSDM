import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_loader import load_orders, build_tensors, STATUS_MAP
from model import TCMNLite

torch.manual_seed(0)


def main():
    df = load_orders("data/orders_expanded.csv")
    trajectories, country_ids, labels, country_to_id = build_tensors(df)

    idx = list(range(len(df)))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=0, stratify=labels)

    model = TCMNLite(num_countries=len(country_to_id))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # class weighting: Cancelled/Pending are under-represented vs Completed
    class_counts = torch.bincount(labels, minlength=3).float()
    class_weights = (1.0 / class_counts) * class_counts.sum() / 3
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    X_train, c_train, y_train = trajectories[train_idx], country_ids[train_idx], labels[train_idx]
    X_test, c_test, y_test = trajectories[test_idx], country_ids[test_idx], labels[test_idx]

    print(f"Train size: {len(train_idx)} | Test size: {len(test_idx)}")

    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train, c_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_test, c_test).argmax(dim=1)
                acc = (preds == y_test).float().mean().item()
            print(f"epoch {epoch+1:3d} | train loss {loss.item():.4f} | test acc {acc:.3f}")

    # --- final evaluation ---
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test, c_test)
        test_preds = test_logits.argmax(dim=1)
        final_acc = (test_preds == y_test).float().mean().item()

    inv_status = {v: k for k, v in STATUS_MAP.items()}
    print(f"\nFinal test accuracy: {final_acc:.3f}")

    # confusion-style breakdown
    for cls_id, cls_name in inv_status.items():
        mask = y_test == cls_id
        if mask.sum() > 0:
            cls_acc = (test_preds[mask] == y_test[mask]).float().mean().item()
            print(f"  {cls_name:10s}: {mask.sum().item():3d} samples | recall {cls_acc:.3f}")

    # --- simple causal-confounder sanity check ---
    # does the model's Cancelled-probability differ meaningfully by country,
    # even after conditioning on a similar price pattern? (quick, not a full ATE estimate)
    print("\nAverage predicted Cancelled-probability by country (confounder check):")
    probs = torch.softmax(model(trajectories, country_ids), dim=1)[:, STATUS_MAP["Cancelled"]]
    for country, cid in country_to_id.items():
        mask = country_ids == cid
        print(f"  {country:10s}: {probs[mask].mean().item():.3f}")

    torch.save(model.state_dict(), "tcmn_lite.pt")
    print("\nSaved model to tcmn_lite.pt")


if __name__ == "__main__":
    main()
