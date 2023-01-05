import optuna
import gymnasium as gym

from model import mlp
from trainer import Trainer


# minibatch_size, num_options, lr, gamma, lam, termination_reg
# model architecture
def objective(trial):
    minibatch_size = trial.suggest_int("minibatch_size", 128, 2048, 128)
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 0.7, 1)
    lam = trial.suggest_float("lam", 0.7, 1)
    n_steps = trial.suggest_int("n_steps", 1, 10)

    vf_arch = [trial.suggest_int(f"vf_arch{i}", 64, 256, 64) for i in range(trial.suggest_int("vf_arch_layers", 1, 3))]
    pi_arch = [trial.suggest_int(f"pi_arch{i}", 64, 256, 64) for i in range(trial.suggest_int("pi_arch_layers", 1, 3))]

    env = gym.make("CartPole-v1")

    trainer = Trainer(env, minibatch_size=minibatch_size, lr=lr, 
                      vf_coef=vf_coef, n_steps=n_steps, gamma=gamma, lam=lam)
    
    trainer.model.pi = mlp(4, pi_arch, trainer.model.act_dim[0])
    trainer.model.vf = mlp(4, vf_arch, 1)

    rets = []

    for i in range(150):
        loss_pi, loss_vf, ep_lens, ep_rets, rollout_time, training_time = trainer.train_one_epoch()

        ep_ret = sum(ep_rets)/len(ep_rets)
        rets.append(ep_ret)

        trial.report(ep_ret, i)
        
    rets.sort(reverse=True)
    print(rets)
    return sum(rets) / 150 # total avg

study = optuna.create_study(study_name="study", sampler=optuna.samplers.TPESampler(), 
                            storage="sqlite:///study.db", direction="maximize")

# study = optuna.load_study(storage="sqlite:///study.db")
study.optimize(objective, n_trials=200)